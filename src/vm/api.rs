use super::execution::ArgCheck;
use super::*;
use crate::ast::Type;
use crate::bytecode::{LustMap, ValueKey, native_fn};
use crate::config::LustConfig;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::mem;
use core::result::Result as CoreResult;
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

/// Formats a doc string as Lust doc comments (`--- text`), one `---` line per
/// source line, indented by `indent`.
#[cfg(feature = "std")]
pub(crate) fn format_doc_comment(doc: &str, indent: &str) -> String {
    let mut out = String::new();
    for line in doc.lines() {
        out.push_str(indent);
        out.push_str("---");
        if !line.is_empty() {
            out.push(' ');
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

#[cfg(feature = "std")]
#[derive(Clone)]
struct VmBuiltinsCache {
    trait_impls: HashMap<(String, String), bool>,
    struct_metadata: HashMap<String, super::RuntimeStructInfo>,
    natives: HashMap<String, Value>,
    globals: HashMap<String, Value>,
    intrinsics: HashMap<usize, crate::jit::Intrinsic>,
}

#[cfg(feature = "std")]
thread_local! {
    static BUILTINS_CACHE: core::cell::RefCell<HashMap<u64, VmBuiltinsCache>> =
        core::cell::RefCell::new(HashMap::new());
}

impl VM {
    pub fn new() -> Self {
        Self::with_config(&LustConfig::default())
    }

    pub fn with_config(config: &LustConfig) -> Self {
        let mut vm = Self {
            current_vm_lookup: super::current_vm_ptr,
            jit: JitState::new(),
            budgets: BudgetState::default(),
            functions: Vec::new(),
            call_meta: Vec::new(),
            natives: HashMap::new(),
            globals: HashMap::new(),
            globals_version: 0,
            jit_cells: crate::jit::JitCells::default(),
            unit_enums: HashMap::new(),
            map_hasher: DefaultHashBuilder::default(),
            call_stack: Vec::new(),
            frame_pool: Vec::new(),
            method_cache: hashbrown::HashMap::new(),
            arg_scratch: Vec::new(),
            max_stack_depth: 1000,
            pending_return_value: None,
            pending_return_dest: None,
            pending_jit_error: None,
            trace_recorder: None,
            nested_loop_exit_ip: None,
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
            #[cfg(feature = "std")]
            exported_type_stubs: Vec::new(),
            last_dispatched_target: None,
            root_frame: Some(CallFrame::new(0, None, 255)),
            active_native_depth: 0,
            active_jit_depth: 0,
        };
        vm.jit.enabled = vm.jit.enabled && config.jit_enabled();

        #[cfg(feature = "std")]
        let cache_key = {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            config.jit_enabled().hash(&mut hasher);
            for m in config.enabled_modules() {
                m.hash(&mut hasher);
            }
            hasher.finish()
        };

        #[cfg(feature = "std")]
        let cached = BUILTINS_CACHE.with(|c| c.borrow().get(&cache_key).cloned());

        #[cfg(feature = "std")]
        if let Some(cached) = cached {
            vm.trait_impls = cached.trait_impls;
            vm.struct_metadata = cached.struct_metadata;
            vm.natives = cached.natives;
            vm.globals = cached.globals;
            vm.jit.intrinsics = cached.intrinsics;
            return vm;
        }

        vm.trait_impls
            .insert(("int".to_string(), "ToString".to_string()), true);
        vm.trait_impls
            .insert(("float".to_string(), "ToString".to_string()), true);
        vm.trait_impls
            .insert(("string".to_string(), "ToString".to_string()), true);
        vm.trait_impls
            .insert(("bool".to_string(), "ToString".to_string()), true);
        vm.register_index_error_layout();
        super::corelib::install_core_builtins(&mut vm);
        #[cfg(feature = "std")]
        for (name, func) in super::stdlib::create_stdlib(config, &vm) {
            vm.register_native(name, func);
        }
        vm.register_jit_intrinsics();

        #[cfg(feature = "std")]
        BUILTINS_CACHE.with(|c| {
            c.borrow_mut().insert(
                cache_key,
                VmBuiltinsCache {
                    trait_impls: vm.trait_impls.clone(),
                    struct_metadata: vm.struct_metadata.clone(),
                    natives: vm.natives.clone(),
                    globals: vm.globals.clone(),
                    intrinsics: vm.jit.intrinsics.clone(),
                },
            );
        });

        vm
    }

    fn register_index_error_layout(&mut self) {
        let field_names = vec![Rc::new("index".to_string()), Rc::new("length".to_string())];
        let int_type = Type::new(TypeKind::Int, crate::ast::Span::dummy());
        let layout = Rc::new(StructLayout::new(
            "IndexError".to_string(),
            field_names,
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![int_type.clone(), int_type],
            vec![None, None],
        ));
        self.struct_metadata
            .insert("IndexError".to_string(), RuntimeStructInfo { layout });
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

    pub(crate) fn observe_value(&mut self, value: &Value) {
        self.cycle_collector.register_value(value);
    }

    pub(crate) fn set_pending_jit_error(&mut self, error: LustError) {
        self.pending_jit_error = Some(error);
    }

    pub(crate) fn observe_value_graph(&mut self, value: &Value) {
        self.cycle_collector.register_graph(value);
    }

    #[inline]
    pub(super) fn maybe_collect_cycles(&mut self) {
        // The trigger check is on every register write; only an actual
        // collection needs the collector taken out to borrow the VM.
        if self.cycle_collector.should_collect() {
            self.collect_cycles();
        }
    }

    // Keep the whole-heap walk (and the collector's move/drop machinery) out
    // of scalar register writes so the small write path can be inlined.
    #[cold]
    #[inline(never)]
    fn collect_cycles(&mut self) {
        let mut collector = mem::take(&mut self.cycle_collector);
        collector.collect(self);
        self.cycle_collector = collector;
    }

    pub fn with_current<F, R>(f: F) -> CoreResult<R, String>
    where
        F: FnOnce(&mut VM) -> CoreResult<R, String>,
    {
        let ptr_opt = super::current_vm_ptr();
        if let Some(ptr) = ptr_opt {
            let vm = unsafe { &mut *ptr };
            f(vm)
        } else {
            Err(
                "No active VM context; this API must be called from a VM-executed native callback"
                    .to_string(),
            )
        }
    }

    pub fn load_functions(&mut self, functions: Vec<Function>) {
        for function in &functions {
            for value in &function.chunk.constants {
                self.observe_value_graph(value);
            }
        }
        self.jit.invalidate_compiled_code();
        self.jit.reset_function_tables(functions.len());
        self.jit_cells.entry_table = self.jit.function_entry_table();
        self.call_meta = functions.iter().map(CallMeta::of).collect();
        self.functions = functions;
    }

    pub fn jit_stats(&self) -> crate::jit::JitStats {
        self.jit.stats()
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
            if let Some(simple) = name.rsplit('.').next()
                && (simple != "IndexError" || name == "IndexError")
            {
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

    pub fn instantiate_struct_ordered(
        &self,
        struct_name: &str,
        ordered_values: Vec<Value>,
    ) -> Result<Value> {
        let info =
            self.struct_metadata
                .get(struct_name)
                .ok_or_else(|| LustError::RuntimeError {
                    message: format!("Unknown struct '{}'", struct_name),
                })?;
        let layout = info.layout.clone();
        Ok(Value::Struct(crate::bytecode::StructObject::new(
            struct_name,
            layout,
            ordered_values,
        )))
    }

    pub(crate) fn array_index_result(&mut self, array: &Value, index: LustInt) -> Result<Value> {
        let values = match array {
            Value::Array(values) => values,
            _ => {
                return Err(LustError::RuntimeError {
                    message: format!("Cannot index {:?}", array.type_of()),
                });
            }
        };
        let values = values.try_borrow().map_err(|_| LustError::RuntimeError {
            message: "Cannot read array while it is mutably borrowed".to_string(),
        })?;
        let length = values.len();
        let value = if index >= 0 {
            values.get(index as usize).cloned()
        } else {
            None
        };
        drop(values);

        if let Some(value) = value {
            self.budgets.charge_value_vec(1)?;
            return Ok(Value::ok(value));
        }

        self.budgets.charge_value_vec(3)?;
        let error = self.instantiate_struct(
            "IndexError",
            vec![
                (Rc::new("index".to_string()), Value::Int(index)),
                (
                    Rc::new("length".to_string()),
                    Value::Int(int_from_usize(length)),
                ),
            ],
        )?;
        Ok(Value::err(error))
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
                    });
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
                .filter(|&(idx, _name)| !filled[idx])
                .map(|(_idx, name)| (**name).clone())
                .collect();
            return Err(LustError::RuntimeError {
                message: format!(
                    "Struct '{}' is missing required field(s): {}",
                    struct_name,
                    missing.join(", ")
                ),
            });
        }

        Ok(Value::Struct(crate::bytecode::StructObject::new(
            struct_name,
            layout,
            ordered,
        )))
    }

    pub fn register_trait_impl(&mut self, type_name: String, trait_name: String) {
        self.trait_impls.insert((type_name, trait_name), true);
    }

    pub fn register_native(&mut self, name: impl Into<String>, value: Value) {
        let name = name.into();
        self.observe_value_graph(&value);
        match value {
            Value::NativeFunction(func) => {
                let host_lookup = self.current_vm_lookup;
                let local_lookup = super::current_vm_ptr as fn() -> Option<*mut VM>;
                let value = if core::ptr::fn_addr_eq(host_lookup, local_lookup) {
                    Value::NativeFunction(func)
                } else {
                    // This registration is executing in an extension's runtime copy.
                    // Resolve the calling VM at invocation time, then expose it to
                    // the extension's VM::with_current for the duration of the call.
                    Value::NativeFunction(native_fn(move |args| {
                        let _context = host_lookup().map(super::CurrentVmGuard::new);
                        func(args)
                    }))
                };
                let cloned = value.clone();
                self.natives.insert(name.clone(), value);
                self.globals.insert(name, cloned);
            }

            other => {
                self.globals.insert(name, other);
            }
        }
        self.globals_version = self.globals_version.wrapping_add(1);
    }

    #[allow(dead_code)]
    pub(crate) fn push_export_prefix(&mut self, crate_name: &str) {
        let sanitized = crate_name.replace('-', "_");
        self.export_prefix_stack.push(sanitized);
    }

    #[allow(dead_code)]
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
        if self
            .exported_natives
            .iter()
            .any(|existing| existing.name == export.name)
        {
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
        let params = export.params.clone();
        let return_type = export.return_type.clone();
        let type_module = self.export_prefix();
        self.push_export_metadata(export);
        let native = Value::NativeFunction(native_fn(move |args| {
            if args.len() != params.len() {
                return Err(format!(
                    "Native expects {} arguments, got {}",
                    params.len(),
                    args.len()
                ));
            }
            VM::with_current(|vm| {
                for (index, (value, param)) in args.iter().zip(&params).enumerate() {
                    if !vm.value_is_export_type(value, param.ty(), type_module.as_deref()) {
                        return Err(format!(
                            "Native argument {} expects {}, got {:?}",
                            index + 1,
                            param.ty(),
                            value.type_of()
                        ));
                    }
                }

                let outcome = func(args)?;
                if let NativeCallResult::Return(value) = &outcome {
                    let expected = if return_type.trim().is_empty() {
                        "()"
                    } else {
                        return_type.as_str()
                    };
                    if !vm.value_is_export_type(value, expected, type_module.as_deref()) {
                        return Err(format!(
                            "Native must return {}, got {:?}",
                            expected,
                            value.type_of()
                        ));
                    }
                }
                Ok(outcome)
            })
        }));
        self.register_native(name, native);
    }

    fn value_is_export_type(&self, value: &Value, expected: &str, module: Option<&str>) -> bool {
        if self.value_is_type(value, expected) {
            return true;
        }
        // Export metadata uses the same module-local type names as the generated
        // extern stubs, while runtime struct and enum values have canonical names.
        module.is_some_and(|module| {
            expected.split('|').map(str::trim).any(|member| {
                !member.contains('.')
                    && !member.contains("::")
                    && self.value_is_type(value, &format!("{module}.{member}"))
            })
        })
    }

    #[cfg(feature = "std")]
    pub fn register_type_stubs(&mut self, stubs: Vec<ModuleStub>) {
        if stubs.is_empty() {
            return;
        }
        self.exported_type_stubs.extend(stubs);
    }

    #[cfg(feature = "std")]
    pub fn exported_type_stubs(&self) -> &[ModuleStub] {
        &self.exported_type_stubs
    }

    #[cfg(feature = "std")]
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
        self.globals_version = self.globals_version.wrapping_add(1);
        #[cfg(feature = "std")]
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
        self.globals_version = self.globals_version.wrapping_add(1);
        self.maybe_collect_cycles();
    }

    fn execute_call_frame(&mut self, frame: Box<CallFrame>) -> Result<Value> {
        let stack_depth_before = self.call_stack.len();
        let saved_pending_return_value = self.pending_return_value.clone();
        let saved_pending_return_dest = self.pending_return_dest;
        let saved_pending_task_signal = self.pending_task_signal.clone();
        let saved_last_task_signal = self.last_task_signal.clone();
        let saved_trace_recorder = self.trace_recorder.take();
        let saved_skip_next_trace_record = self.skip_next_trace_record;
        let saved_call_until_depth = self.call_until_depth;
        self.skip_next_trace_record = false;
        self.call_until_depth = Some(stack_depth_before);
        let func_idx = frame.function_idx;
        self.call_stack.push(frame);
        #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
        let result = if self.jit.enabled && saved_trace_recorder.is_none() {
            match self.run_compiled_function(func_idx) {
                Ok(Some(v)) => Ok(v),
                Ok(None) => {
                    if let Some(target_depth) = self.call_until_depth
                        && self.call_stack.len() == target_depth
                        && let Some(return_value) = self.pending_return_value.take()
                    {
                        self.call_until_depth = None;
                        Ok(return_value)
                    } else {
                        self.run()
                    }
                }
                Err(e) => Err(e),
            }
        } else {
            self.run()
        };
        #[cfg(not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        let result = self.run();
        if self.trace_recorder.is_some() {
            self.abandon_trace_recording();
        }
        self.trace_recorder = saved_trace_recorder;
        self.skip_next_trace_record = saved_skip_next_trace_record;
        self.call_until_depth = saved_call_until_depth;
        match result {
            Ok(v) => Ok(v),
            Err(e) => {
                let annotated = self.annotate_runtime_error(e);
                self.call_stack.truncate(stack_depth_before);
                self.pending_return_value = saved_pending_return_value;
                self.pending_return_dest = saved_pending_return_dest;
                self.pending_task_signal = saved_pending_task_signal;
                self.last_task_signal = saved_last_task_signal;
                Err(annotated)
            }
        }
    }

    fn call_function_index(&mut self, func_idx: usize, args: Vec<Value>) -> Result<Value> {
        let frame = self.make_call_frame(func_idx, None, args, Vec::new())?;
        self.execute_call_frame(frame)
    }

    pub fn call(&mut self, function_name: &str, args: Vec<Value>) -> Result<Value> {
        let func_idx = self
            .functions
            .iter()
            .position(|f| f.name == function_name)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Function not found: {}", function_name),
            })?;
        self.call_function_index(func_idx, args)
    }

    pub(crate) fn functions(&self) -> &[Function] {
        &self.functions
    }

    pub(crate) fn last_dispatched_target(&self) -> Option<(usize, String)> {
        self.last_dispatched_target.clone()
    }

    pub(crate) fn current_frame_target(&self) -> Option<(usize, &str)> {
        self.call_stack.last().and_then(|frame| {
            let idx = frame.function_idx;
            self.functions.get(idx).map(|f| (idx, f.name.as_str()))
        })
    }

    pub(crate) fn call_function_index_slice(
        &mut self,
        func_idx: usize,
        args: &[Value],
    ) -> Result<Value> {
        #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
        if self.call_stack.is_empty() && self.jit.enabled && self.trace_recorder.is_none() {
            let entry_addr = self.jit.function_entry(func_idx);
            if entry_addr != 0 {
                let (param_count, register_count) = (
                    self.functions[func_idx].param_count as usize,
                    self.functions[func_idx].register_count,
                );
                if args.len() == param_count {
                    if let Some(mut root_frame) = self.root_frame.take() {
                        root_frame.registers[..param_count].clone_from_slice(args);
                        let registers_ptr = root_frame.registers.as_mut_ptr();
                        let vm_ptr = self as *mut VM;

                        let marker = 0u8;
                        let sp = &marker as *const u8 as usize;
                        let limit = sp.saturating_sub(crate::jit::NATIVE_STACK_RESERVE);
                        let outer_limit =
                            core::mem::replace(&mut self.jit_cells.stack_limit, limit);

                        self.jit.record_native_entry();
                        self.pending_jit_error = None;
                        self.jit_cells.exit_info = usize::MAX;

                        type NativeEntry = unsafe extern "C" fn(
                            *mut Value,
                            *mut VM,
                            *const Function,
                            *mut (),
                        ) -> i32;
                        let entry_fn: NativeEntry = unsafe { core::mem::transmute(entry_addr) };
                        let result = unsafe {
                            entry_fn(
                                registers_ptr,
                                vm_ptr,
                                core::ptr::null(),
                                core::ptr::null_mut(),
                            )
                        };

                        self.jit_cells.stack_limit = outer_limit;

                        if (crate::jit::FUNCTION_RETURN_BASE..crate::jit::NATIVE_RETURNED)
                            .contains(&result)
                        {
                            let reg = (result - crate::jit::FUNCTION_RETURN_BASE) as usize;
                            let return_value = root_frame.registers[reg].fast_clone();
                            root_frame.reset(0, None, register_count);
                            self.root_frame = Some(root_frame);
                            return Ok(return_value);
                        }
                        root_frame.reset(0, None, register_count);
                        self.root_frame = Some(root_frame);
                    }
                }
            }
        }
        let frame = self.make_call_frame_slice(func_idx, None, args, ArgCheck::Shallow)?;
        self.execute_call_frame(frame)
    }

    #[inline]
    pub(crate) fn has_native_entry(&self, func_idx: usize) -> bool {
        #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.call_stack.is_empty()
                && self.jit.enabled
                && self.trace_recorder.is_none()
                && self.root_frame.is_some()
                && self.jit.function_entry(func_idx) != 0
        }
        #[cfg(not(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64"))))]
        {
            false
        }
    }

    #[cfg(all(feature = "std", any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub(crate) fn call_direct_fast_with<F: FnOnce(&mut [Value]) -> usize>(
        &mut self,
        func_idx: usize,
        write_args: F,
    ) -> Option<Result<Value>> {
        if self.call_stack.is_empty() && self.jit.enabled && self.trace_recorder.is_none() {
            let entry_addr = self.jit.function_entry(func_idx);
            if entry_addr != 0 {
                let (param_count, register_count) = (
                    self.functions[func_idx].param_count as usize,
                    self.functions[func_idx].register_count,
                );
                if let Some(mut root_frame) = self.root_frame.take() {
                    let arg_count = write_args(&mut root_frame.registers);
                    if arg_count == param_count {
                        let registers_ptr = root_frame.registers.as_mut_ptr();
                        let vm_ptr = self as *mut VM;

                        let marker = 0u8;
                        let sp = &marker as *const u8 as usize;
                        let limit = sp.saturating_sub(crate::jit::NATIVE_STACK_RESERVE);
                        let outer_limit =
                            core::mem::replace(&mut self.jit_cells.stack_limit, limit);

                        self.jit.record_native_entry();
                        self.pending_jit_error = None;
                        self.jit_cells.exit_info = usize::MAX;

                        type NativeEntry = unsafe extern "C" fn(
                            *mut Value,
                            *mut VM,
                            *const Function,
                            *mut (),
                        ) -> i32;
                        let entry_fn: NativeEntry = unsafe { core::mem::transmute(entry_addr) };
                        let result = unsafe {
                            entry_fn(
                                registers_ptr,
                                vm_ptr,
                                core::ptr::null(),
                                core::ptr::null_mut(),
                            )
                        };

                        self.jit_cells.stack_limit = outer_limit;

                        if (crate::jit::FUNCTION_RETURN_BASE..crate::jit::NATIVE_RETURNED)
                            .contains(&result)
                        {
                            let reg = (result - crate::jit::FUNCTION_RETURN_BASE) as usize;
                            let return_value = root_frame.registers[reg].fast_clone();
                            root_frame.reset(0, None, register_count);
                            self.root_frame = Some(root_frame);
                            return Some(Ok(return_value));
                        }

                        let fallback_res = self
                            .make_call_frame_slice(
                                func_idx,
                                None,
                                &root_frame.registers[..param_count],
                                ArgCheck::Shallow,
                            )
                            .and_then(|frame| self.execute_call_frame(frame));
                        root_frame.reset(0, None, register_count);
                        self.root_frame = Some(root_frame);
                        return Some(fallback_res);
                    } else {
                        self.root_frame = Some(root_frame);
                    }
                }
            }
        }
        None
    }


    pub(crate) fn call_retained_target(
        &mut self,
        func_idx: usize,
        expected_name: &str,
        args: Vec<Value>,
    ) -> Result<Value> {
        let function = self
            .functions
            .get(func_idx)
            .ok_or_else(|| LustError::RuntimeError {
                message: format!("Function index out of bounds: {}", func_idx),
            })?;
        if function.name != expected_name {
            return Err(LustError::RuntimeError {
                message: format!(
                    "Function table staled or mismatched: index {} has name '{}', expected '{}'",
                    func_idx, function.name, expected_name
                ),
            });
        }
        if self
            .last_dispatched_target
            .as_ref()
            .map(|(idx, name)| (*idx, name.as_str()))
            != Some((func_idx, expected_name))
        {
            self.last_dispatched_target = Some((func_idx, expected_name.to_string()));
        }
        self.call_function_index(func_idx, args)
    }

    pub(crate) fn call_retained_target_slice(
        &mut self,
        func_idx: usize,
        expected_name: &str,
        args: &[Value],
    ) -> Result<Value> {
        let is_cached = self
            .last_dispatched_target
            .as_ref()
            .map(|(idx, name)| *idx == func_idx && name.as_str() == expected_name)
            .unwrap_or(false);
        if !is_cached {
            let function = self
                .functions
                .get(func_idx)
                .ok_or_else(|| LustError::RuntimeError {
                    message: format!("Function index out of bounds: {}", func_idx),
                })?;
            if function.name != expected_name {
                return Err(LustError::RuntimeError {
                    message: format!(
                        "Function table staled or mismatched: index {} has name '{}', expected '{}'",
                        func_idx, function.name, expected_name
                    ),
                });
            }
            self.last_dispatched_target = Some((func_idx, expected_name.to_string()));
        }
        self.call_function_index_slice(func_idx, args)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Span, TypeKind};
    use crate::typechecker::FunctionSignature;

    #[test]
    fn typed_numeric_instructions_match_generic_semantics() {
        use crate::number::NumericType;
        let operations = [
            Instruction::Add(2, 0, 1),
            Instruction::Sub(2, 0, 1),
            Instruction::Mul(2, 0, 1),
            Instruction::Div(2, 0, 1),
            Instruction::Mod(2, 0, 1),
            Instruction::Neg(2, 0),
            Instruction::Eq(2, 0, 1),
            Instruction::Ne(2, 0, 1),
            Instruction::Lt(2, 0, 1),
            Instruction::Le(2, 0, 1),
            Instruction::Gt(2, 0, 1),
            Instruction::Ge(2, 0, 1),
        ];
        for kind in [NumericType::Int, NumericType::Float] {
            let inputs = match kind {
                NumericType::Int => vec![
                    (Value::Int(7), Value::Int(3)),
                    (Value::Int(-7), Value::Int(3)),
                    (Value::Int(0), Value::Int(0)),
                ],
                NumericType::Float => vec![
                    (Value::Float(7.5), Value::Float(2.0)),
                    (Value::Float(-7.5), Value::Float(2.0)),
                    (Value::Float(LustFloat::NAN), Value::Float(1.0)),
                    (
                        Value::Float(LustFloat::INFINITY),
                        Value::Float(LustFloat::INFINITY),
                    ),
                    (Value::Float(0.0), Value::Float(-0.0)),
                ],
            };
            for op in operations {
                let typed = op.specialize_numeric(kind);
                assert_eq!(typed.numeric_specialization().unwrap().0, op);
                assert_eq!(typed.defined_register(), Some(2));
                assert!(typed.reads_register(0));
                assert!(!typed.reads_register(2));
                let mut functions = Vec::new();
                for (name, instruction) in [("generic", op), ("typed", typed)] {
                    let mut function = Function::new(name, 2, false);
                    function.set_register_count(3);
                    function.chunk.emit(instruction, 1);
                    function.chunk.emit(Instruction::Return(2), 1);
                    functions.push(function);
                }
                let mut vm = VM::new();
                vm.load_functions(functions);
                for (left, right) in &inputs {
                    let expected = vm.call("generic", vec![left.clone(), right.clone()]);
                    let actual = vm.call("typed", vec![left.clone(), right.clone()]);
                    match (actual, expected) {
                        (Ok(Value::Float(a)), Ok(Value::Float(b))) => {
                            assert!(
                                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                                "{typed}: {a} != {b}"
                            );
                        }
                        (Ok(a), Ok(b)) => assert_eq!(a, b, "{typed}"),
                        (Err(_), Err(_)) => {}
                        (a, b) => panic!("{typed}: {a:?} != {b:?}"),
                    }
                }
                assert!(
                    vm.call("typed", vec![Value::Bool(true), Value::Bool(false)])
                        .is_err()
                );
                let wrong_numeric = match kind {
                    NumericType::Int => Value::Float(1.0),
                    NumericType::Float => Value::Int(1),
                };
                assert!(
                    vm.call("typed", vec![wrong_numeric.clone(), wrong_numeric])
                        .is_err()
                );
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn embedded_numeric_types_are_module_specific() {
        for low_memory in [false, true] {
            let config = LustConfig::default()
                .with_low_memory_mode(low_memory)
                .with_minimal_runtime_types(low_memory);
            let mut program = crate::EmbeddedProgram::builder()
                .with_config(config)
                .module(
                    "integers",
                    "function add(n: int, m: int): int\n    return n + m\nend\n",
                )
                .module(
                    "floats",
                    "function add(n: float, m: float): float\n    return n + m\nend\n",
                )
                .module("main", "use integers\nuse floats\n")
                .entry_module("main")
                .compile()
                .unwrap();
            for (name, expected) in [
                ("integers.add", Instruction::AddInt(2, 0, 1)),
                ("floats.add", Instruction::AddFloat(2, 0, 1)),
            ] {
                let function = program
                    .vm_mut()
                    .functions
                    .iter()
                    .find(|f| f.name == name)
                    .unwrap();
                assert!(
                    function.chunk.instructions.contains(&expected),
                    "{}",
                    function.disassemble()
                );
            }
            assert_eq!(
                program
                    .call_raw("integers.add", vec![Value::Int(2), Value::Int(3)])
                    .unwrap(),
                Value::Int(5)
            );
            assert_eq!(
                program
                    .call_raw("floats.add", vec![Value::Float(2.5), Value::Float(3.0)])
                    .unwrap(),
                Value::Float(5.5)
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn compiled_typed_loops_execute_and_reject_host_type_violations() {
        let source = r#"
struct Counter
    value: int
end
function counter(): Counter
    return Counter { value = 2 }
end
function sum(n: int, counter: Counter): int
    local total: int = 0
    for i = 1, n do
        total += counter.value
    end
    return total
end
function floats(): float
    local total: float = 0.0
    for i = 1.0, 100.0, 1.0 do
        total += i
    end
    return total
end
function option_sum(n: int): int
    local total: int = 0
    local counter: int = 0
    for i = 1, n do
        counter += 1
        local opt: Option<int> = Option.Some(counter)
        if opt is Some(value) then
            total += value
        end
    end
    return total
end
"#;
        for jit in [false, true] {
            let mut config = LustConfig::default();
            config.set_jit_enabled(jit);
            let mut program = crate::EmbeddedProgram::builder()
                .with_config(config)
                .module("main", source)
                .entry_module("main")
                .compile()
                .unwrap();
            let counter = program.call_raw("main.counter", vec![]).unwrap();
            assert_eq!(
                program
                    .call_raw("main.sum", vec![Value::Int(100), counter.clone()])
                    .unwrap(),
                Value::Int(200)
            );
            assert_eq!(
                program.call_raw("main.floats", vec![]).unwrap(),
                Value::Float(5050.0)
            );
            assert_eq!(
                program
                    .call_raw("main.option_sum", vec![Value::Int(100)])
                    .unwrap(),
                Value::Int(5050)
            );
            if jit && cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) {
                assert!(program.jit_stats().root_traces_compiled >= 2);
                assert!(program.jit_stats().native_trace_entries >= 2);
            }
            counter
                .struct_set_field("value", Value::Float(2.5))
                .unwrap();
            let error = program
                .call_raw("main.sum", vec![Value::Int(100), counter])
                .unwrap_err();
            assert!(
                error.to_string().contains("expected int operands"),
                "{error}"
            );
        }
    }

    fn typed_function(
        name: &str,
        params: Vec<Type>,
        return_type: Type,
        returned: Value,
    ) -> Function {
        let mut function = Function::new(name, params.len() as u8, false);
        function.set_register_count((params.len() + 1) as u8);
        let result = function.chunk.add_constant(returned);
        function
            .chunk
            .emit(Instruction::LoadConst(params.len() as u8, result), 1);
        function
            .chunk
            .emit(Instruction::Return(params.len() as u8), 1);
        function.set_signature(FunctionSignature {
            params,
            return_type,
            is_method: false,
            type_params: Vec::new(),
            trait_bounds: Vec::new(),
        });
        function
    }

    fn ty(kind: TypeKind) -> Type {
        Type::new(kind, Span::dummy())
    }

    #[test]
    fn public_calls_validate_argument_types() {
        let function = typed_function(
            "typed",
            vec![ty(TypeKind::Int)],
            ty(TypeKind::Int),
            Value::Int(7),
        );
        let mut vm = VM::new();
        vm.load_functions(vec![function]);

        let error = vm.call("typed", vec![Value::Bool(true)]).unwrap_err();
        assert!(error.to_string().contains("argument 1 expects int"));
        assert!(matches!(
            vm.call("typed", vec![Value::Int(1)]),
            Ok(Value::Int(7))
        ));
    }

    #[test]
    fn numeric_comparisons_preserve_integer_precision_and_nan_semantics() {
        let cases = [
            (
                Value::Int(LustInt::MAX - 1),
                Value::Int(LustInt::MAX),
                [true, true, false, false],
            ),
            (
                Value::Int(LustInt::MIN + 1),
                Value::Int(LustInt::MIN),
                [false, false, true, true],
            ),
            (
                Value::Int(LustInt::MAX),
                Value::Int(LustInt::MAX),
                [false, true, false, true],
            ),
            (
                Value::Float(1.5),
                Value::Float(2.5),
                [true, true, false, false],
            ),
            (
                Value::Float(-0.0),
                Value::Float(0.0),
                [false, true, false, true],
            ),
            (Value::Float(LustFloat::NAN), Value::Float(1.0), [false; 4]),
            (Value::Float(1.0), Value::Float(LustFloat::NAN), [false; 4]),
            (Value::Int(2), Value::Float(2.5), [true, true, false, false]),
            (Value::Float(2.5), Value::Int(2), [false, false, true, true]),
        ];
        for (index, instruction) in [
            Instruction::Lt(2, 0, 1),
            Instruction::Le(2, 0, 1),
            Instruction::Gt(2, 0, 1),
            Instruction::Ge(2, 0, 1),
        ]
        .into_iter()
        .enumerate()
        {
            let mut function = Function::new("compare", 2, false);
            function.set_register_count(3);
            function.chunk.emit(instruction, 1);
            function.chunk.emit(Instruction::Return(2), 1);
            let mut vm = VM::new();
            vm.jit.enabled = false;
            vm.load_functions(vec![function]);
            for (left, right, expected) in &cases {
                let result = vm
                    .call("compare", vec![left.clone(), right.clone()])
                    .unwrap();
                assert_eq!(
                    result,
                    Value::Bool(expected[index]),
                    "{instruction:?}: {left:?}, {right:?}"
                );
            }
            let error = vm
                .call("compare", vec![Value::Bool(true), Value::Int(1)])
                .unwrap_err();
            assert!(error.to_string().contains("Cannot compare"));
        }
    }

    #[test]
    fn regression_while_loop_conditional_accumulator() {
        let source = r#"
function count(): int
    local c: int = 0
    local i: int = 0
    while i < 1000 do
        if i % 2 == 0 then
            c = c + 1
        end
        i = i + 1
    end
    return c
end
"#;
        for jit in [false, true] {
            let mut config = LustConfig::default();
            config.set_jit_enabled(jit);
            let mut program = crate::EmbeddedProgram::builder()
                .with_config(config)
                .module("main", source)
                .entry_module("main")
                .compile()
                .unwrap();
            let result = program.call_raw("main.count", vec![]).unwrap();
            assert_eq!(result, Value::Int(500), "jit: {jit}");
        }
    }

    #[test]
    fn standalone_vm_can_construct_index_errors() {
        let mut vm = VM::new();
        let array = Value::array(vec![Value::Int(1)]);

        let result = vm
            .array_index_result(&array, 2)
            .expect("built-in IndexError layout");
        let error = match result.as_enum() {
            Some(("Result", "Err", Some([error]))) => error,
            other => panic!("expected Result.Err(IndexError), got {other:?}"),
        };
        assert!(matches!(
            error.struct_get_field("index"),
            Some(Value::Int(2))
        ));
        assert!(matches!(
            error.struct_get_field("length"),
            Some(Value::Int(1))
        ));
    }

    #[test]
    fn lua_value_arguments_accept_underlying_vm_values() {
        let function = typed_function(
            "lua_arg",
            vec![ty(TypeKind::Named("LuaValue".to_string()))],
            ty(TypeKind::Int),
            Value::Int(7),
        );
        let lua_table = Value::Struct(crate::bytecode::StructObject::new(
            "LuaTable",
            Rc::new(StructLayout::new(
                "LuaTable".to_string(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )),
            Vec::new(),
        ));
        let mut vm = VM::new();
        vm.load_functions(vec![function]);

        assert!(matches!(
            vm.call("lua_arg", vec![lua_table]),
            Ok(Value::Int(7))
        ));
    }

    #[test]
    fn lua_functions_apply_lua_arity_rules() {
        let lua_value = ty(TypeKind::Named("LuaValue".to_string()));
        let function = typed_function(
            "lua_function",
            vec![lua_value.clone(), lua_value.clone()],
            lua_value,
            Value::Int(1),
        );
        let mut vm = VM::new();
        vm.load_functions(vec![function]);

        assert!(vm.call("lua_function", vec![Value::Int(1)]).is_ok());
        assert!(
            vm.call(
                "lua_function",
                vec![Value::Int(1), Value::Int(2), Value::Int(3)]
            )
            .is_ok()
        );

        let zero_arg = typed_function(
            "zero_arg_lua_function",
            Vec::new(),
            ty(TypeKind::Array(Box::new(ty(TypeKind::Named(
                "LuaValue".to_string(),
            ))))),
            Value::array(Vec::new()),
        );
        vm.load_functions(vec![zero_arg]);
        assert!(
            vm.call("zero_arg_lua_function", vec![Value::Int(1)])
                .is_ok()
        );

        let empty_return = typed_function(
            "empty_lua_return",
            Vec::new(),
            ty(TypeKind::Array(Box::new(ty(TypeKind::Named(
                "LuaValue".to_string(),
            ))))),
            Value::Nil,
        );
        vm.load_functions(vec![empty_return]);
        assert!(matches!(
            vm.call("empty_lua_return", Vec::new()),
            Ok(Value::Nil)
        ));
    }

    #[test]
    fn call_value_validates_arity_before_building_a_frame() {
        let function = typed_function(
            "typed",
            vec![ty(TypeKind::Int)],
            ty(TypeKind::Int),
            Value::Int(7),
        );
        let mut vm = VM::new();
        vm.load_functions(vec![function]);

        let error = vm.call_value(&Value::Function(0), Vec::new()).unwrap_err();
        assert!(error.to_string().contains("expects 1 arguments, got 0"));
        assert!(vm.call_stack.is_empty());
    }

    #[test]
    fn declared_return_types_are_validated_before_frame_exit() {
        let function = typed_function(
            "bad_return",
            Vec::new(),
            ty(TypeKind::Int),
            Value::Bool(true),
        );
        let mut vm = VM::new();
        vm.load_functions(vec![function]);

        let error = vm.call("bad_return", Vec::new()).unwrap_err();
        assert!(error.to_string().contains("must return int"));
        assert!(vm.call_stack.is_empty());
    }

    #[test]
    fn argument_validation_checks_container_elements() {
        let function = typed_function(
            "array_arg",
            vec![ty(TypeKind::Array(Box::new(ty(TypeKind::Int))))],
            ty(TypeKind::Int),
            Value::Int(7),
        );
        let mut vm = VM::new();
        vm.load_functions(vec![function]);

        let error = vm
            .call("array_arg", vec![Value::array(vec![Value::Bool(true)])])
            .unwrap_err();
        assert!(error.to_string().contains("expects Array<int>"));
    }

    #[test]
    fn missing_native_context_diagnostic_is_not_task_specific() {
        let error = VM::with_current(|_| Ok(())).unwrap_err();
        assert!(error.contains("No active VM context"));
        assert!(error.contains("native callback"));
        assert!(!error.contains("task API"));
    }

    #[cfg(feature = "std")]
    #[test]
    fn native_callbacks_bridge_runtime_contexts_and_restore_the_previous_vm() {
        use core::cell::Cell;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        std::thread_local! {
            static HOST_VM: Cell<Option<*mut VM>> = const { Cell::new(None) };
        }
        fn host_lookup() -> Option<*mut VM> {
            HOST_VM.with(Cell::get)
        }
        struct HostContext;
        impl Drop for HostContext {
            fn drop(&mut self) {
                HOST_VM.with(|slot| slot.set(None));
            }
        }
        let _host_context = HostContext;

        let mut registration_vm = VM::new();
        registration_vm.current_vm_lookup = host_lookup;
        registration_vm.register_exported_native(
            NativeExport::new("current_marker", Vec::new(), "int"),
            |_| VM::with_current(|vm| Ok(vm.get_global("marker").unwrap().into())),
        );
        registration_vm.register_native(
            "panic_in_callback",
            Value::NativeFunction(native_fn(|_| {
                VM::with_current::<_, ()>(|_| panic!("callback panic"))?;
                unreachable!()
            })),
        );
        let Value::NativeFunction(callback) = registration_vm.get_global("current_marker").unwrap()
        else {
            panic!("expected native callback");
        };
        let Value::NativeFunction(panicking) =
            registration_vm.get_global("panic_in_callback").unwrap()
        else {
            panic!("expected native callback");
        };

        // Model the host and extension having independent thread-local stacks.
        // The callback must use the invoking VM, not the registration-time VM.
        let mut previous_vm = VM::new();
        let previous_ptr = &mut previous_vm as *mut VM;
        let _previous_context = crate::vm::CurrentVmGuard::new(previous_ptr);
        for marker in [7, 42] {
            let mut host_vm = VM::new();
            host_vm.set_global("marker", Value::Int(marker));
            HOST_VM.with(|slot| slot.set(Some(&mut host_vm)));
            assert!(
                matches!(callback(&[]), Ok(NativeCallResult::Return(Value::Int(n))) if n == marker)
            );
            assert_eq!(crate::vm::current_vm_ptr(), Some(previous_ptr));
            assert!(callback(&[Value::Nil]).is_err());
            assert_eq!(crate::vm::current_vm_ptr(), Some(previous_ptr));
            assert!(catch_unwind(AssertUnwindSafe(|| panicking(&[]))).is_err());
            assert_eq!(crate::vm::current_vm_ptr(), Some(previous_ptr));
            HOST_VM.with(|slot| slot.set(None));
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn exported_nominal_types_resolve_in_the_extension_module() {
        use crate::embed::{ExternRegistry, StructBuilder};

        let mut vm = VM::new();
        vm.push_export_prefix("extension");
        let mut registry = ExternRegistry::new();
        registry.add_struct(StructBuilder::new("Factor").finish());
        registry.add_struct(StructBuilder::new("other.Factor").finish());
        registry.register_with_vm(&mut vm);
        for (name, nominal_type) in [("factor", "Factor"), ("operation", "Operation")] {
            vm.register_exported_native(
                NativeExport::new(
                    name,
                    vec![NativeExportParam::new("value", nominal_type)],
                    nominal_type,
                ),
                |args| Ok(args[0].clone().into()),
            );
        }
        vm.register_exported_native(
            NativeExport::new("bad_factor", Vec::new(), "Factor"),
            |_| {
                VM::with_current(|vm| {
                    vm.instantiate_struct("other.Factor", Vec::new())
                        .map(NativeCallResult::Return)
                        .map_err(|error| error.to_string())
                })
            },
        );
        vm.pop_export_prefix();

        let factor = vm
            .instantiate_struct("extension.Factor", Vec::new())
            .unwrap();
        let other_factor = vm.instantiate_struct("other.Factor", Vec::new()).unwrap();
        for (name, valid, invalid) in [
            ("extension.factor", factor, other_factor),
            (
                "extension.operation",
                Value::enum_variant("extension.Operation", "Double", Vec::new()),
                Value::enum_variant("other.Operation", "Double", Vec::new()),
            ),
        ] {
            let callback = vm.get_global(name).unwrap();
            assert_eq!(
                vm.call_value(&callback, vec![valid.clone()]).unwrap(),
                valid
            );
            let error = vm.call_value(&callback, vec![invalid]).unwrap_err();
            assert!(error.to_string().contains("Native argument 1 expects"));
        }
        let bad_factor = vm.get_global("extension.bad_factor").unwrap();
        let error = vm.call_value(&bad_factor, Vec::new()).unwrap_err();
        assert!(error.to_string().contains("Native must return Factor"));
    }

    #[test]
    fn exported_function_types_accept_closures_without_explicit_return() {
        let mut program = crate::EmbeddedProgram::builder()
            .module(
                "main",
                r#"
                    extern
                        function call_it(fn: function())
                    end

                    function main()
                        call_it(function()
                            local unused = 1
                        end)
                    end
                "#,
            )
            .entry_module("main")
            .compile()
            .expect("compile");

        let called = std::rc::Rc::new(std::cell::Cell::new(false));
        let flag = called.clone();
        program.vm_mut().register_exported_native(
            crate::NativeExport::new(
                "main.call_it",
                vec![crate::NativeExportParam::new("fn", "function()")],
                "()",
            ),
            move |args| {
                match &args[0] {
                    Value::Function(_) | Value::Closure { .. } | Value::NativeFunction(_) => {}
                    other => {
                        return Err(format!("expected a callable, got {:?}", other.type_of()));
                    }
                }
                crate::VM::with_current(|vm| {
                    vm.call_value(&args[0], Vec::new())
                        .map(|_| ())
                        .map_err(|error| error.to_string())
                })?;
                flag.set(true);
                Ok(NativeCallResult::Return(Value::Nil))
            },
        );

        program.call_raw("main.main", Vec::new()).expect("call");
        assert!(called.get(), "callback should have been invoked");
    }

    #[test]
    fn exported_function_types_reject_closures_that_return_a_value() {
        let result = crate::EmbeddedProgram::builder()
            .module(
                "main",
                r#"
                    extern
                        function call_it(fn: function())
                    end

                    function main()
                        call_it(function()
                            return 5
                        end)
                    end
                "#,
            )
            .entry_module("main")
            .compile();

        let error = match result {
            Ok(_) => panic!("expected compile error"),
            Err(err) => err.to_string(),
        };
        assert!(
            error.contains("expected 'function(): ()', got 'function(): int'"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn exported_function_types_match_explicit_signatures() {
        let mut program = crate::EmbeddedProgram::builder()
            .module(
                "main",
                r#"
                    extern
                        function call_it(fn: function(int): int): int
                    end

                    function main(): int
                        return call_it(function(n: int): int
                            return n * 2
                        end)
                    end
                "#,
            )
            .entry_module("main")
            .compile()
            .expect("compile");

        program.vm_mut().register_exported_native(
            crate::NativeExport::new(
                "main.call_it",
                vec![crate::NativeExportParam::new("fn", "function(int): int")],
                "int",
            ),
            |args| {
                crate::VM::with_current(|vm| {
                    vm.call_value(&args[0], vec![Value::Int(21)])
                        .map(NativeCallResult::Return)
                        .map_err(|error| error.to_string())
                })
            },
        );

        let result = program.call_raw("main.main", Vec::new()).expect("call");
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn exported_native_results_are_checked_against_metadata() {
        let mut vm = VM::new();
        vm.register_exported_native(NativeExport::new("bad_native", Vec::new(), "int"), |_| {
            Ok(NativeCallResult::Return(Value::Bool(true)))
        });
        let native = vm.get_global("bad_native").unwrap();

        let error = vm.call_value(&native, Vec::new()).unwrap_err();
        assert!(error.to_string().contains("Native must return int"));
    }

    #[test]
    fn failed_public_call_unwinds_its_frame() {
        let mut failing = Function::new("failing", 0, false);
        failing.set_register_count(3);
        let one = failing.chunk.add_constant(Value::Int(1));
        let zero = failing.chunk.add_constant(Value::Int(0));
        failing.chunk.emit(Instruction::LoadConst(0, one), 1);
        failing.chunk.emit(Instruction::LoadConst(1, zero), 1);
        failing.chunk.emit(Instruction::Div(2, 0, 1), 1);
        failing.chunk.emit(Instruction::Return(2), 1);

        let mut succeeding = Function::new("succeeding", 0, false);
        succeeding.set_register_count(1);
        let result = succeeding.chunk.add_constant(Value::Int(7));
        succeeding.chunk.emit(Instruction::LoadConst(0, result), 1);
        succeeding.chunk.emit(Instruction::Return(0), 1);

        let mut vm = VM::new();
        vm.load_functions(vec![failing, succeeding]);
        vm.trace_recorder = Some(TraceRecorder::new(99, 7, 32));
        vm.skip_next_trace_record = true;

        assert!(vm.call("failing", Vec::new()).is_err());
        assert!(vm.call_stack.is_empty());
        assert!(vm.trace_recorder.is_some());
        assert!(vm.skip_next_trace_record);
        assert!(matches!(
            vm.call("succeeding", Vec::new()),
            Ok(Value::Int(7))
        ));
    }

    #[test]
    fn vm_call_positive_and_shared_execution_path_integrity() {
        let mut add_fn = Function::new("add", 2, false);
        add_fn.set_register_count(3);
        add_fn.chunk.emit(Instruction::Add(2, 0, 1), 1);
        add_fn.chunk.emit(Instruction::Return(2), 1);

        let mut vm = VM::new();
        vm.load_functions(vec![add_fn]);

        let result = vm
            .call("add", vec![Value::Int(10), Value::Int(32)])
            .expect("ordinary VM::call must succeed");
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn vm_retained_target_and_call_nested_error_cleanup_and_subsequent_call() {
        let mut child_ok = Function::new("child_ok", 1, false);
        child_ok.set_register_count(2);
        let ten = child_ok.chunk.add_constant(Value::Int(10));
        child_ok.chunk.emit(Instruction::LoadConst(1, ten), 1);
        child_ok.chunk.emit(Instruction::Add(0, 0, 1), 1);
        child_ok.chunk.emit(Instruction::Return(0), 1);

        let mut child_fail = Function::new("child_fail", 1, false);
        child_fail.set_register_count(2);
        let zero = child_fail.chunk.add_constant(Value::Int(0));
        child_fail.chunk.emit(Instruction::LoadConst(1, zero), 1);
        child_fail.chunk.emit(Instruction::Div(0, 0, 1), 1);
        child_fail.chunk.emit(Instruction::Return(0), 1);

        let mut parent_caller = Function::new("parent_caller", 1, false);
        parent_caller.set_register_count(4);
        let child_ok_val = parent_caller.chunk.add_constant(Value::Function(0));
        parent_caller
            .chunk
            .emit(Instruction::LoadConst(1, child_ok_val), 1);
        parent_caller.chunk.emit(Instruction::Call(1, 0, 1, 2), 1);
        parent_caller.chunk.emit(Instruction::Return(2), 1);

        let mut parent_failing = Function::new("parent_failing", 1, false);
        parent_failing.set_register_count(4);
        let child_fail_val = parent_failing.chunk.add_constant(Value::Function(1));
        parent_failing
            .chunk
            .emit(Instruction::LoadConst(1, child_fail_val), 1);
        parent_failing.chunk.emit(Instruction::Call(1, 0, 1, 2), 1);
        parent_failing.chunk.emit(Instruction::Return(2), 1);

        let mut vm = VM::new();
        vm.load_functions(vec![child_ok, child_fail, parent_caller, parent_failing]);

        // 1. Nested call via VM::call
        let res = vm
            .call("parent_caller", vec![Value::Int(5)])
            .expect("nested call succeeds");
        assert_eq!(res, Value::Int(15));
        assert!(vm.call_stack.is_empty(), "stack restored after success");

        // 2. Nested call via call_retained_target
        let res2 = vm
            .call_retained_target(2, "parent_caller", vec![Value::Int(20)])
            .expect("retained-target nested call succeeds");
        assert_eq!(res2, Value::Int(30));
        assert!(vm.call_stack.is_empty(), "stack restored after success");
        assert_eq!(
            vm.last_dispatched_target(),
            Some((2, "parent_caller".to_string()))
        );

        // 3. Stale / mismatched retained target checks
        let err_oob = vm
            .call_retained_target(999, "parent_caller", vec![Value::Int(0)])
            .expect_err("out of bounds must fail");
        assert!(err_oob.to_string().contains("out of bounds"));

        let err_name = vm
            .call_retained_target(2, "wrong_name", vec![Value::Int(0)])
            .expect_err("mismatched name must fail");
        assert!(err_name.to_string().contains("mismatched"));

        // 4. Runtime error in nested call via call_retained_target
        let err = vm
            .call_retained_target(3, "parent_failing", vec![Value::Int(10)])
            .expect_err("nested division by zero must error");
        assert!(
            err.to_string().to_lowercase().contains("division by zero"),
            "unexpected error message: {}",
            err
        );

        // Verify stack/frame state restoration after error
        assert!(
            vm.call_stack.is_empty(),
            "call stack must be completely restored/unwound after error"
        );
        assert!(vm.pending_return_value.is_none());
        assert!(vm.pending_return_dest.is_none());
        assert!(vm.pending_task_signal.is_none());
        assert!(vm.last_task_signal.is_none());

        // 5. Subsequent valid call on the same VM after error
        let res_subsequent = vm
            .call_retained_target(0, "child_ok", vec![Value::Int(50)])
            .expect("subsequent call on same VM after error must succeed");
        assert_eq!(res_subsequent, Value::Int(60));
        assert_eq!(
            vm.last_dispatched_target(),
            Some((0, "child_ok".to_string()))
        );

        let res_subsequent_call = vm
            .call("parent_caller", vec![Value::Int(100)])
            .expect("subsequent public call on same VM after error must succeed");
        assert_eq!(res_subsequent_call, Value::Int(110));
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_guard_exit_resumes_at_bailout_ip() {
        use crate::jit::TraceId;
        use crate::jit::trace::{Trace, TraceOp};

        let mut function = Function::new("guard_exit", 0, false);
        function.set_register_count(2);
        let initial = function.chunk.add_constant(Value::Int(11));
        let wrong_path = function.chunk.add_constant(Value::Int(22));
        let bailout_path = function.chunk.add_constant(Value::Int(99));
        function.chunk.emit(Instruction::LoadBool(0, false), 1);
        function.chunk.emit(Instruction::LoadConst(1, initial), 1);
        function.chunk.emit(Instruction::Jump(-3), 1);
        function
            .chunk
            .emit(Instruction::LoadConst(1, wrong_path), 1);
        function.chunk.emit(Instruction::Return(1), 1);
        function
            .chunk
            .emit(Instruction::LoadConst(1, bailout_path), 1);
        function.chunk.emit(Instruction::Return(1), 1);

        let trace = Trace {
            function_idx: 0,
            start_ip: 0,
            is_function: false,
            frame_may_own: true,
            entry_scalars: Vec::new(),
            alias_params: 0,
            borrowed_registers: Vec::new(),
            preamble: Vec::new(),
            ops: vec![TraceOp::GuardLoopContinue {
                condition_register: 0,
                expect_truthy: true,
                bailout_ip: 5,
            }],
            postamble: Vec::new(),
            inputs: vec![0],
            outputs: Vec::new(),
        };
        let compiled = JitCompiler::new()
            .compile_trace(&trace, TraceId(0), Vec::new())
            .unwrap();

        let mut vm = VM::new();
        vm.load_functions(vec![function]);
        vm.jit.store_root_trace(0, 0, compiled);

        assert!(matches!(
            vm.call("guard_exit", Vec::new()),
            Ok(Value::Int(99))
        ));
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn counting_loop() -> Function {
        let mut function = Function::new("count", 1, false);
        function.set_register_count(5);
        let zero = function.chunk.add_constant(Value::Int(0));
        let one = function.chunk.add_constant(Value::Int(1));
        function.chunk.emit(Instruction::LoadConst(1, zero), 1);
        function.chunk.emit(Instruction::LoadConst(2, one), 1);
        function.chunk.emit(Instruction::Lt(3, 1, 0), 1);
        function.chunk.emit(Instruction::JumpIfNot(3, 3), 1);
        function.chunk.emit(Instruction::Add(4, 1, 2), 1);
        function.chunk.emit(Instruction::Move(1, 4), 1);
        function.chunk.emit(Instruction::Jump(-5), 1);
        function.chunk.emit(Instruction::Return(1), 1);
        function
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn recursive_sum(function_idx: usize) -> Function {
        let mut function = Function::new("sum_down", 1, false);
        function.set_register_count(9);
        let zero = function.chunk.add_constant(Value::Int(0));
        let one = function.chunk.add_constant(Value::Int(1));
        let recursive = function.chunk.add_constant(Value::Function(function_idx));
        function.chunk.emit(Instruction::LoadConst(1, zero), 1);
        function.chunk.emit(Instruction::Eq(2, 0, 1), 1);
        function.chunk.emit(Instruction::JumpIfNot(2, 1), 1);
        function.chunk.emit(Instruction::Return(1), 1);
        function.chunk.emit(Instruction::LoadConst(3, one), 1);
        function.chunk.emit(Instruction::Sub(4, 0, 3), 1);
        function.chunk.emit(Instruction::LoadConst(5, recursive), 1);
        function.chunk.emit(Instruction::Move(6, 4), 1);
        function.chunk.emit(Instruction::Call(5, 6, 1, 7), 1);
        function.chunk.emit(Instruction::Add(8, 0, 7), 1);
        function.chunk.emit(Instruction::Return(8), 1);
        function
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn loop_calling_recursive_sum() -> Function {
        let mut function = Function::new("sum_in_loop", 1, false);
        function.set_register_count(11);
        let zero = function.chunk.add_constant(Value::Int(0));
        let one = function.chunk.add_constant(Value::Int(1));
        let callee = function.chunk.add_constant(Value::Function(0));
        let five = function.chunk.add_constant(Value::Int(5));
        function.chunk.emit(Instruction::LoadConst(1, zero), 1);
        function.chunk.emit(Instruction::LoadConst(2, zero), 1);
        function.chunk.emit(Instruction::LoadConst(3, one), 1);
        function.chunk.emit(Instruction::LoadConst(4, callee), 1);
        function.chunk.emit(Instruction::LoadConst(5, five), 1);
        function.chunk.emit(Instruction::Lt(6, 1, 0), 1);
        function.chunk.emit(Instruction::JumpIfNot(6, 7), 1);
        function.chunk.emit(Instruction::Move(7, 5), 1);
        function.chunk.emit(Instruction::Call(4, 7, 1, 8), 1);
        function.chunk.emit(Instruction::Add(9, 2, 8), 1);
        function.chunk.emit(Instruction::Move(2, 9), 1);
        function.chunk.emit(Instruction::Add(10, 1, 3), 1);
        function.chunk.emit(Instruction::Move(1, 10), 1);
        function.chunk.emit(Instruction::Jump(-9), 1);
        function.chunk.emit(Instruction::Return(2), 1);
        function
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn nested_counting_loop() -> Function {
        let mut function = Function::new("nested_count", 0, false);
        function.set_register_count(9);
        let zero = function.chunk.add_constant(Value::Int(0));
        let one = function.chunk.add_constant(Value::Int(1));
        let sixteen = function.chunk.add_constant(Value::Int(16));
        function.chunk.emit(Instruction::LoadConst(0, zero), 1);
        function.chunk.emit(Instruction::LoadConst(1, one), 1);
        function.chunk.emit(Instruction::LoadConst(2, sixteen), 1);
        function.chunk.emit(Instruction::LoadConst(3, one), 1);
        function.chunk.emit(Instruction::Le(4, 1, 2), 1);
        function.chunk.emit(Instruction::JumpIfNot(4, 11), 1);
        function.chunk.emit(Instruction::Move(5, 3), 1);
        function.chunk.emit(Instruction::Le(6, 5, 3), 1);
        function.chunk.emit(Instruction::JumpIfNot(6, 5), 1);
        function.chunk.emit(Instruction::Add(7, 0, 3), 1);
        function.chunk.emit(Instruction::Move(0, 7), 1);
        function.chunk.emit(Instruction::Add(8, 5, 3), 1);
        function.chunk.emit(Instruction::Move(5, 8), 1);
        function.chunk.emit(Instruction::Jump(-7), 1);
        function.chunk.emit(Instruction::Add(7, 1, 3), 1);
        function.chunk.emit(Instruction::Move(1, 7), 1);
        function.chunk.emit(Instruction::Jump(-13), 1);
        function.chunk.emit(Instruction::Return(0), 1);
        function
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn hot_loop_compiles_executes_and_reuses_its_trace() {
        let mut vm = VM::new();
        vm.load_functions(vec![counting_loop()]);

        assert!(matches!(
            vm.call("count", vec![Value::Int(100)]),
            Ok(Value::Int(100))
        ));
        let first = vm.jit_stats();
        assert_eq!(first.root_traces_compiled, 1);
        assert!(first.native_trace_entries > 0);

        assert!(matches!(
            vm.call("count", vec![Value::Int(20)]),
            Ok(Value::Int(20))
        ));
        let second = vm.jit_stats();
        assert_eq!(second.root_traces_compiled, 1);
        assert!(second.native_trace_entries > first.native_trace_entries);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn short_hot_loop_retries_recording_on_a_later_call() {
        let mut vm = VM::new();
        vm.load_functions(vec![counting_loop()]);

        assert!(matches!(
            vm.call("count", vec![Value::Int(5)]),
            Ok(Value::Int(5))
        ));
        let first = vm.jit_stats();
        assert_eq!(first.recordings_started, 1);
        assert_eq!(first.recordings_aborted, 1);
        assert_eq!(first.root_traces_compiled, 0);

        assert!(matches!(
            vm.call("count", vec![Value::Int(5)]),
            Ok(Value::Int(5))
        ));
        let second = vm.jit_stats();
        assert_eq!(second.root_traces_compiled, 1);
        assert!(second.native_trace_entries > 0);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn pure_recursion_is_profiled_without_becoming_a_loop_trace() {
        let mut vm = VM::new();
        vm.load_functions(vec![recursive_sum(0)]);

        assert!(matches!(
            vm.call("sum_down", vec![Value::Int(10)]),
            Ok(Value::Int(55))
        ));
        let stats = vm.jit_stats();
        assert_eq!(stats.function_calls, 11);
        assert_eq!(stats.recursive_calls, 10);
        assert_eq!(stats.recordings_started, 0);
        assert_eq!(stats.root_traces_compiled, 0);
        assert_eq!(stats.native_trace_entries, 0);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn hot_loop_can_trace_across_an_opaque_recursive_call() {
        let mut vm = VM::new();
        vm.load_functions(vec![recursive_sum(0), loop_calling_recursive_sum()]);

        assert!(matches!(
            vm.call("sum_in_loop", vec![Value::Int(100)]),
            Ok(Value::Int(1500))
        ));
        let stats = vm.jit_stats();
        assert_eq!(stats.root_traces_compiled, 1);
        assert!(stats.native_trace_entries > 0);
        assert!(stats.recursive_calls > 0);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn nested_loop_recording_does_not_mix_enclosing_backedges() {
        let mut vm = VM::new();
        vm.load_functions(vec![nested_counting_loop()]);

        assert!(matches!(
            vm.call("nested_count", Vec::new()),
            Ok(Value::Int(16))
        ));
        assert!(vm.jit_stats().recordings_aborted > 0);
    }
}
