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
        if config.use_mir() {
            vm.jit.enabled = false;
        } else {
            vm.jit.enabled = vm.jit.enabled && config.jit_enabled();
        }

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
    pub(crate) fn record_struct_fast_path(&mut self) {
        self.jit.record_struct_fast_path();
    }

    #[inline]
    pub(crate) fn record_struct_slow_path(&mut self) {
        self.jit.record_struct_slow_path();
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

    #[inline]
    pub fn cycle_collector_tracked_count(&self) -> usize {
        self.cycle_collector.tracked_count()
    }

    pub fn force_collect_cycles(&mut self) {
        self.collect_cycles();
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

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_fast_path_ordered_and_scrambled() {
        let code = r#"
struct Point
    x: int
    y: int
end

function make_ordered(n: int): int
    local i: int = 0
    local sum: int = 0
    while i < n do
        local p: Point = Point { x = i, y = i * 2 }
        sum = sum + p.x + p.y
        i = i + 1
    end
    return sum
end

function make_scrambled(n: int): int
    local i: int = 0
    local sum: int = 0
    while i < n do
        local p: Point = Point { y = i * 2, x = i }
        sum = sum + p.x + p.y
        i = i + 1
    end
    return sum
end
"#;
        let mut program = crate::EmbeddedProgram::builder()
            .module("main", code)
            .entry_module("main")
            .compile()
            .unwrap();

        // 1. Ordered fields hit the fast path
        let res_ordered = program.call_raw("main.make_ordered", vec![Value::Int(200)]).unwrap();
        assert_eq!(res_ordered, Value::Int(59700));
        let stats_ordered = program.vm().jit_stats();
        assert!(stats_ordered.struct_fast_path_hits > 0, "fast path must be hit for ordered fields");

        // 2. Scrambled fields fall back to slow path, but compute exact correct result
        let prev_fallbacks = stats_ordered.struct_slow_path_fallbacks;
        let res_scrambled = program.call_raw("main.make_scrambled", vec![Value::Int(200)]).unwrap();
        assert_eq!(res_scrambled, Value::Int(59700));
        let stats_scrambled = program.vm().jit_stats();
        assert!(
            stats_scrambled.struct_slow_path_fallbacks > prev_fallbacks,
            "slow path must be taken for reordered/scrambled fields"
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_value_types_and_nested() {
        let code = r#"
struct Inner
    id: int
    active: bool
end

struct Outer
    inner: Inner
    score: float
end

function build_nested(i: int): Outer
    local in_obj: Inner = Inner { id = i, active = true }
    return Outer { inner = in_obj, score = 1.5 }
end

function make_nested(n: int): float
    local i: int = 0
    local acc: float = 0.0
    while i < n do
        local out_obj: Outer = build_nested(i)
        if out_obj.inner.active then
            acc = acc + out_obj.score
        end
        i = i + 1
    end
    return acc
end
"#;
        let mut program = crate::EmbeddedProgram::builder()
            .module("main", code)
            .entry_module("main")
            .compile()
            .unwrap();

        let res = program.call_raw("main.make_nested", vec![Value::Int(150)]).unwrap();
        assert_eq!(res, Value::Float(225.0));
        let stats = program.vm().jit_stats();
        assert!(stats.struct_fast_path_hits > 0, "nested structs must hit fast path");
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_destination_aliasing_argument() {
        use crate::ast::{Type, TypeKind};
        use crate::bytecode::value::{FieldStorage, StructLayout};
        use crate::ast::Span;
        use std::rc::Rc;
        let mut vm = VM::new();

        // Register struct Pair { a: int, b: int }
        let int_ty = Type::new(TypeKind::Int, Span::dummy());
        let layout = Rc::new(StructLayout::new(
            "Pair".to_string(),
            vec![Rc::new("a".to_string()), Rc::new("b".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![int_ty.clone(), int_ty.clone()],
            vec![None, None],
        ));
        vm.struct_metadata.insert(
            "Pair".to_string(),
            crate::vm::RuntimeStructInfo { layout: layout.clone() },
        );

        // Bytecode function where dest register (R1) aliases first_field (R1):
        // R1 = 42, R2 = 99
        // NewStruct R1, "Pair", ["a", "b"], R1..R2, count=2
        // Return R1
        let mut function = Function::new("alias_test", 0, false);
        function.set_register_count(4);
        let pair_name = function.chunk.add_constant(Value::string("Pair"));
        let a_name = function.chunk.add_constant(Value::string("a"));
        let _b_name = function.chunk.add_constant(Value::string("b"));
        let val_42 = function.chunk.add_constant(Value::Int(42));
        let val_99 = function.chunk.add_constant(Value::Int(99));

        function.chunk.emit(Instruction::LoadConst(1, val_42), 1);
        function.chunk.emit(Instruction::LoadConst(2, val_99), 1);
        function.chunk.emit(Instruction::NewStruct(1, pair_name, a_name, 1, 2), 1);
        function.chunk.emit(Instruction::Return(1), 1);

        vm.load_functions(vec![function]);
        let result = vm.call("alias_test", vec![]).unwrap();
        if let Value::Struct(obj) = result {
            let fields = obj.fields.borrow();
            assert_eq!(fields[0], Value::Int(42), "first field must preserve value even when dest aliases argument");
            assert_eq!(fields[1], Value::Int(99), "second field must preserve value");
        } else {
            panic!("expected Struct, got {:?}", result);
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_metadata_mutation_and_invalidation() {
        use crate::ast::{Span, Type, TypeKind};
        use crate::bytecode::value::{FieldStorage, StructLayout, jit_new_struct_safe};
        use std::rc::Rc;

        let mut vm = VM::new();
        let int_ty = Type::new(TypeKind::Int, Span::dummy());

        // 1. Initial 2-field layout: Data { u: int, v: int }
        let initial_layout = Rc::new(StructLayout::new(
            "Data".to_string(),
            vec![Rc::new("u".to_string()), Rc::new("v".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![int_ty.clone(), int_ty.clone()],
            vec![None, None],
        ));
        vm.struct_metadata.insert(
            "Data".to_string(),
            crate::vm::RuntimeStructInfo { layout: initial_layout.clone() },
        );

        let struct_name = b"Data";
        let field_u = b"u";
        let field_v = b"v";
        let field_ptrs: [*const u8; 2] = [field_u.as_ptr(), field_v.as_ptr()];
        let field_lens: [usize; 2] = [1, 1];
        let values = [Value::Int(10), Value::Int(20)];
        let mut out = Value::Nil;

        // Fast path hit with matching valid layout
        let ok = unsafe {
            jit_new_struct_safe(
                &mut vm as *mut VM,
                struct_name.as_ptr(),
                struct_name.len(),
                field_ptrs.as_ptr(),
                field_lens.as_ptr(),
                values.as_ptr(),
                2,
                &mut out as *mut Value,
            )
        };
        assert_eq!(ok, 1);
        assert_eq!(vm.jit_stats().struct_fast_path_hits, 1);
        assert_eq!(vm.jit_stats().struct_slow_path_fallbacks, 0);

        // Registration also supports short aliases for qualified layout names.
        // The constructed object's name must be the requested metadata key.
        vm.struct_metadata.insert(
            "Alias".to_string(),
            crate::vm::RuntimeStructInfo { layout: initial_layout },
        );
        let alias = b"Alias";
        let alias_ok = unsafe {
            jit_new_struct_safe(
                &mut vm as *mut VM, alias.as_ptr(), alias.len(), field_ptrs.as_ptr(),
                field_lens.as_ptr(), values.as_ptr(), 2, &mut out as *mut Value,
            )
        };
        assert_eq!(alias_ok, 1);
        let expected = vm.instantiate_struct("Alias", vec![
            (Rc::new("u".to_string()), Value::Int(10)),
            (Rc::new("v".to_string()), Value::Int(20)),
        ]).unwrap();
        if let (Value::Struct(actual), Value::Struct(reference)) = (&out, expected) {
            assert_eq!(actual.name, reference.name);
            assert_eq!(*actual.fields.borrow(), *reference.fields.borrow());
        } else {
            panic!("expected structures");
        }
        assert_eq!(vm.jit_stats().struct_slow_path_fallbacks, 1);

        // 2. Mutate metadata: replace "Data" with a 3-field layout (u, v, w)
        let mutated_layout = Rc::new(StructLayout::new(
            "Data".to_string(),
            vec![Rc::new("u".to_string()), Rc::new("v".to_string()), Rc::new("w".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong, FieldStorage::Strong],
            vec![int_ty.clone(), int_ty.clone(), int_ty],
            vec![None, None, None],
        ));
        vm.struct_metadata.insert(
            "Data".to_string(),
            crate::vm::RuntimeStructInfo { layout: mutated_layout },
        );

        // Calling with the old 2 fields: fast path must reject due to field count mismatch,
        // and fall back to slow path (which returns 0 because missing required field 'w')
        let ok_mutated = unsafe {
            jit_new_struct_safe(
                &mut vm as *mut VM,
                struct_name.as_ptr(),
                struct_name.len(),
                field_ptrs.as_ptr(),
                field_lens.as_ptr(),
                values.as_ptr(),
                2,
                &mut out as *mut Value,
            )
        };
        assert_eq!(ok_mutated, 0);
        assert_eq!(vm.jit_stats().struct_slow_path_fallbacks, 2);

        // 3. Mutate metadata: completely remove "Data"
        vm.struct_metadata.remove("Data");
        let ok_missing = unsafe {
            jit_new_struct_safe(
                &mut vm as *mut VM,
                struct_name.as_ptr(),
                struct_name.len(),
                field_ptrs.as_ptr(),
                field_lens.as_ptr(),
                values.as_ptr(),
                2,
                &mut out as *mut Value,
            )
        };
        assert_eq!(ok_missing, 0);
        assert_eq!(vm.jit_stats().struct_slow_path_fallbacks, 3);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_destination_aliasing_argument_real_jit() {
        use crate::ast::{Span, Type, TypeKind};
        use crate::bytecode::value::{FieldStorage, StructLayout};
        use std::rc::Rc;

        let mut vm = VM::new();

        let int_ty = Type::new(TypeKind::Int, Span::dummy());
        let layout = Rc::new(StructLayout::new(
            "Pair".to_string(),
            vec![Rc::new("a".to_string()), Rc::new("b".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![int_ty.clone(), int_ty.clone()],
            vec![None, None],
        ));
        vm.struct_metadata.insert(
            "Pair".to_string(),
            crate::vm::RuntimeStructInfo { layout: layout.clone() },
        );

        // Function 1: Dest (R1) aliases first field (R1)
        let mut f1 = Function::new("alias_first", 0, false);
        f1.set_register_count(4);
        let pair_name = f1.chunk.add_constant(Value::string("Pair"));
        let a_name = f1.chunk.add_constant(Value::string("a"));
        let _b_name = f1.chunk.add_constant(Value::string("b"));
        let val_42 = f1.chunk.add_constant(Value::Int(42));
        let val_99 = f1.chunk.add_constant(Value::Int(99));

        f1.chunk.emit(Instruction::LoadConst(1, val_42), 1);
        f1.chunk.emit(Instruction::LoadConst(2, val_99), 1);
        f1.chunk.emit(Instruction::NewStruct(1, pair_name, a_name, 1, 2), 1);
        f1.chunk.emit(Instruction::Return(1), 1);

        // Function 2: Dest (R2) aliases second field (R2)
        let mut f2 = Function::new("alias_second", 0, false);
        f2.set_register_count(4);
        let pair_name2 = f2.chunk.add_constant(Value::string("Pair"));
        let a_name2 = f2.chunk.add_constant(Value::string("a"));
        let _b_name2 = f2.chunk.add_constant(Value::string("b"));
        let val_42_2 = f2.chunk.add_constant(Value::Int(42));
        let val_99_2 = f2.chunk.add_constant(Value::Int(99));

        f2.chunk.emit(Instruction::LoadConst(1, val_42_2), 1);
        f2.chunk.emit(Instruction::LoadConst(2, val_99_2), 1);
        f2.chunk.emit(Instruction::NewStruct(2, pair_name2, a_name2, 1, 2), 1);
        f2.chunk.emit(Instruction::Return(2), 1);

        vm.load_functions(vec![f1, f2]);

        // Call alias_first repeatedly to cross FUNCTION_HOT_THRESHOLD (15) and compile/run natively in JIT
        for _ in 0..25 {
            let res = vm.call("alias_first", vec![]).unwrap();
            if let Value::Struct(obj) = res {
                let fields = obj.fields.borrow();
                assert_eq!(fields[0], Value::Int(42));
                assert_eq!(fields[1], Value::Int(99));
            } else {
                panic!("expected Struct, got {:?}", res);
            }
        }

        let stats1 = vm.jit_stats();
        assert!(stats1.native_trace_entries > 0, "native JIT execution must be entered");
        assert!(stats1.struct_fast_path_hits > 0, "fast path must be hit in JIT when dest aliases first field");

        let prev_hits = stats1.struct_fast_path_hits;
        let prev_native = stats1.native_trace_entries;

        // Call alias_second repeatedly to compile/run natively in JIT
        for _ in 0..25 {
            let res = vm.call("alias_second", vec![]).unwrap();
            if let Value::Struct(obj) = res {
                let fields = obj.fields.borrow();
                assert_eq!(fields[0], Value::Int(42));
                assert_eq!(fields[1], Value::Int(99));
            } else {
                panic!("expected Struct, got {:?}", res);
            }
        }

        let stats2 = vm.jit_stats();
        assert!(stats2.native_trace_entries > prev_native, "native JIT execution must continue");
        assert!(stats2.struct_fast_path_hits > prev_hits, "fast path must be hit in JIT when dest aliases second field");
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_empty_fields_real_jit() {
        let code = r#"
struct Empty
end

function make_empty(n: int): int
    local i: int = 0
    while i < n do
        local e: Empty = Empty {}
        i = i + 1
    end
    return i
end
"#;
        let mut program = crate::EmbeddedProgram::builder()
            .module("main", code)
            .entry_module("main")
            .compile()
            .unwrap();

        let res = program.call_raw("main.make_empty", vec![Value::Int(150)]).unwrap();
        assert_eq!(res, Value::Int(150));
        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "JIT trace must be compiled for empty struct loop");
        assert!(stats.native_trace_entries > 0, "native trace must be entered");
        assert!(stats.struct_fast_path_hits > 0, "empty struct must hit fast path in JIT");
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_weak_option_fallback_real_jit() {
        let code = r#"
struct Target
    val: int
end

struct WeakHolder
    target: ref Target
end

function make_weak_loop(n: int): int
    local t: Target = Target { val = 42 }
    local i: int = 0
    local sum: int = 0
    while i < n do
        local h: WeakHolder = WeakHolder { target = t }
        sum = sum + 1
        i = i + 1
    end
    return sum
end
"#;
        let mut program = crate::EmbeddedProgram::builder()
            .module("main", code)
            .entry_module("main")
            .compile()
            .unwrap();

        let res = program.call_raw("main.make_weak_loop", vec![Value::Int(150)]).unwrap();
        assert_eq!(res, Value::Int(150));
        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "JIT trace must compile for weak holder loop");
        assert!(stats.native_trace_entries > 0, "native trace must execute");
        assert!(stats.struct_slow_path_fallbacks > 0, "structs with ref/weak fields must fall back to slow path");
        assert_eq!(stats.struct_fast_path_hits, 0, "fast path must not be taken for weak fields");
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_gc_observation_and_ownership() {
        let code = r#"
struct Payload
    id: int
end

struct Boxed
    inner: Payload
end

function loop_alloc(n: int): int
    local i: int = 0
    local sum: int = 0
    while i < n do
        local b: Boxed = Boxed { inner = Payload { id = i } }
        sum = sum + b.inner.id
        i = i + 1
    end
    return sum
end

function make_boxed(id: int): Boxed
    return Boxed { inner = Payload { id = id } }
end
"#;
        let mut program = crate::EmbeddedProgram::builder()
            .module("main", code)
            .entry_module("main")
            .compile()
            .unwrap();

        let initial_containers = program.vm().cycle_collector_tracked_count();

        let res = program.call_raw("main.loop_alloc", vec![Value::Int(150)]).unwrap();
        assert_eq!(res, Value::Int(11175));

        let stats = program.vm().jit_stats();
        assert!(stats.native_trace_entries > 0, "native JIT trace must execute");
        assert!(stats.struct_fast_path_hits > 0, "structs must hit fast path");

        // 1. Verify live Weak handle for JIT-allocated acyclic struct:
        // Demonstrates that acyclic structs are reclaimed immediately via standard Rc refcounting,
        // unlike cyclic structures which require cycle collection (tested in jit_struct_cyclic_gc_live_weak_upgrade_real_jit).
        let boxed = program.call_raw("main.make_boxed", vec![Value::Int(77)]).unwrap();
        let weak_boxed = if let Value::Struct(ref obj) = boxed {
            std::rc::Rc::downgrade(obj)
        } else {
            panic!("expected Struct");
        };
        assert!(weak_boxed.upgrade().is_some(), "struct must be alive while root handle is held");
        drop(boxed);
        assert!(
            weak_boxed.upgrade().is_none(),
            "acyclic struct is deallocated immediately upon dropping root via Rc refcount"
        );

        // 2. GC cycle collector container tracking:
        // Note: tracked_count reflects registered weak handles in CycleCollector::containers.
        let post_containers = program.vm().cycle_collector_tracked_count();
        assert!(post_containers > initial_containers, "cycle collector must observe new containers");

        // Force cycle collection: dead weak entries from swept containers are purged
        program.vm_mut().force_collect_cycles();
        let after_gc_containers = program.vm().cycle_collector_tracked_count();
        assert!(
            after_gc_containers < post_containers,
            "dead weak entries must be swept by the cycle collector (was {}, after {})",
            post_containers,
            after_gc_containers
        );
        assert_eq!(
            after_gc_containers,
            initial_containers,
            "all dead struct container entries must be swept back to initial baseline"
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_weak_canonicalization_and_release() {
        let code = r#"
struct Target
    val: int
end

struct WeakHolder
    target: ref Target
end

function read_target(h: WeakHolder): int
    if h.target is Some(t) then
        return t.val
    end
    return -1
end

function make_dropped(): WeakHolder
    local inner: Target = Target { val = 99 }
    local holder: WeakHolder = WeakHolder { target = inner }
    -- when function returns, local 'inner' is dropped, so weak ref becomes dead!
    return holder
end

function loop_weak_deref(n: int): int
    local t: Target = Target { val = 7 }
    local i: int = 0
    local sum: int = 0
    while i < n do
        local h: WeakHolder = WeakHolder { target = t }
        if h.target is Some(target_ref) then
            sum = sum + target_ref.val
        end
        i = i + 1
    end
    return sum
end
"#;
        let mut program = crate::EmbeddedProgram::builder()
            .module("main", code)
            .entry_module("main")
            .compile()
            .unwrap();

        // 1. Test live weak reference in loop with JIT
        let sum = program.call_raw("main.loop_weak_deref", vec![Value::Int(50)]).unwrap();
        assert_eq!(sum, Value::Int(350)); // 50 * 7

        let stats = program.vm().jit_stats();
        assert!(stats.struct_slow_path_fallbacks > 0, "weak structs must fall back to slow path");
        assert_eq!(stats.struct_fast_path_hits, 0, "fast path must not be taken for weak structs");

        // 2. Test dropped target: weak reference becomes dead / Option::None
        let dropped_holder = program.call_raw("main.make_dropped", vec![]).unwrap();
        let read_val = program.call_raw("main.read_target", vec![dropped_holder]).unwrap();
        assert_eq!(read_val, Value::Int(-1), "dropped weak target must resolve to Option::None");

        // 3. Directly inspect underlying StructObject canonicalization
        // Field storage must contain Option::Some(WeakStruct)
        if let Value::Struct(obj) = program.call_raw("main.make_dropped", vec![]).unwrap() {
            let fields = obj.fields.borrow();
            assert_eq!(fields.len(), 1);
            if let Value::Enum(enum_obj) = &fields[0] {
                assert_eq!(enum_obj.enum_name, "Option");
                assert_eq!(enum_obj.variant, "Some");
                if let Some(payloads) = &enum_obj.values {
                    assert_eq!(payloads.len(), 1);
                    if let Value::WeakStruct(weak_ref) = &payloads[0] {
                        assert!(weak_ref.upgrade().is_none(), "underlying WeakStruct must be dropped/expired");
                    } else {
                        panic!("expected Value::WeakStruct in payload, got {:?}", payloads[0]);
                    }
                } else {
                    panic!("expected payload in Option::Some");
                }
            } else {
                panic!("expected Value::Enum for canonicalized weak field, got {:?}", fields[0]);
            }
        } else {
            panic!("expected StructObject");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_trace_recorder_new_struct_aliasing_and_bailout() {
        use crate::ast::{Span, Type, TypeKind};
        use crate::bytecode::value::{FieldStorage, StructLayout};
        use std::rc::Rc;

        let mut vm = VM::new();

        let any_ty = Type::new(TypeKind::Unknown, Span::dummy());
        let layout = Rc::new(StructLayout::new(
            "Pair".to_string(),
            vec![Rc::new("a".to_string()), Rc::new("b".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![any_ty.clone(), any_ty.clone()],
            vec![None, None],
        ));
        vm.struct_metadata.insert(
            "Pair".to_string(),
            crate::vm::RuntimeStructInfo { layout: layout.clone() },
        );

        // Case 1: Loop trace where NewStruct destination (R1) aliases its first input register (R1).
        // TraceRecorder must NOT abort on instruction.reads_register(dest) because NewStruct
        // preserves its inputs.
        // R0: loop limit (50)
        // R1: dest & first field reg
        // R2: second field reg (constant 99)
        // R3: loop counter 'i' (starts at 0)
        // R4: step 1
        // R5: comparison (i < limit)
        let mut f1 = Function::new("trace_loop_alias_first", 1, false);
        f1.set_register_count(6);

        let pair_name = f1.chunk.add_constant(Value::string("Pair"));
        let a_name = f1.chunk.add_constant(Value::string("a"));
        let _b_name = f1.chunk.add_constant(Value::string("b"));
        let val_42 = f1.chunk.add_constant(Value::Int(42));
        let val_99 = f1.chunk.add_constant(Value::Int(99));
        let val_0 = f1.chunk.add_constant(Value::Int(0));
        let val_1 = f1.chunk.add_constant(Value::Int(1));

        // IP 0: LoadConst R1 = 42
        f1.chunk.emit(Instruction::LoadConst(1, val_42), 1);
        // IP 1: LoadConst R2 = 99
        f1.chunk.emit(Instruction::LoadConst(2, val_99), 1);
        // IP 2: LoadConst R3 = 0
        f1.chunk.emit(Instruction::LoadConst(3, val_0), 1);
        // IP 3: LoadConst R4 = 1
        f1.chunk.emit(Instruction::LoadConst(4, val_1), 1);

        // Loop header at IP 4:
        // IP 4: LtInt R5 = R3 < R0
        f1.chunk.emit(Instruction::LtInt(5, 3, 0), 2);
        // IP 5: JumpIfNot R5, +3 (to IP 9 Return: next_ip is 6, 6 + 3 = 9)
        f1.chunk.emit(Instruction::JumpIfNot(5, 3), 2);
        // IP 6: NewStruct dest=1, first_field_reg=1, count=2 (DEST 1 ALIASES FIELD 1!)
        f1.chunk.emit(Instruction::NewStruct(1, pair_name, a_name, 1, 2), 2);
        // IP 7: AddInt R3 = R3 + R4
        f1.chunk.emit(Instruction::AddInt(3, 3, 4), 2);
        // IP 8: Jump back to IP 4 (offset: 4 - 9 = -5)
        f1.chunk.emit(Instruction::Jump(-5), 2);

        // IP 9: Return R1
        f1.chunk.emit(Instruction::Return(1), 3);

        // Case 2: Loop trace with NewStruct destination aliasing second input (R2)
        // and mid-trace bailout (bailout guard when i reaches 25).
        // R0: loop limit (50)
        // R1: first field reg (constant 42)
        // R2: second field reg & dest of NewStruct
        // R3: loop counter 'i' (starts at 0)
        // R4: step 1
        // R5: comparison (i < limit)
        // R6: bailout limit (25)
        // R7: bailout condition (i < 25)
        // Case 1b: Loop trace with NewStruct dest=1 aliasing first field reg=1,
        // with scalar update each iteration: R1 = R3, R2 = 99.
        // Directly verifies the scalar value of the overlapping field (field 0) at loop exit.
        let mut f1_scalar = Function::new("trace_loop_alias_first_scalar", 1, false);
        f1_scalar.set_register_count(6);
        let pair_name1s = f1_scalar.chunk.add_constant(Value::string("Pair"));
        let a_name1s = f1_scalar.chunk.add_constant(Value::string("a"));
        let _b_name1s = f1_scalar.chunk.add_constant(Value::string("b"));
        let val_99_1s = f1_scalar.chunk.add_constant(Value::Int(99));
        let val_0_1s = f1_scalar.chunk.add_constant(Value::Int(0));
        let val_1_1s = f1_scalar.chunk.add_constant(Value::Int(1));

        // IP 0: LoadConst R2 = 99
        f1_scalar.chunk.emit(Instruction::LoadConst(2, val_99_1s), 1);
        // IP 1: LoadConst R3 = 0
        f1_scalar.chunk.emit(Instruction::LoadConst(3, val_0_1s), 1);
        // IP 2: LoadConst R4 = 1
        f1_scalar.chunk.emit(Instruction::LoadConst(4, val_1_1s), 1);

        // Loop header at IP 3:
        // IP 3: LtInt R5 = R3 < R0
        f1_scalar.chunk.emit(Instruction::LtInt(5, 3, 0), 2);
        // IP 4: JumpIfNot R5, +4 (to IP 9 Return R1: 5 + 4 = 9)
        f1_scalar.chunk.emit(Instruction::JumpIfNot(5, 4), 2);
        // IP 5: Move R1 = R3 (current counter value into R1)
        f1_scalar.chunk.emit(Instruction::Move(1, 3), 2);
        // IP 6: NewStruct dest=1, first_field_reg=1, count=2 (DEST 1 ALIASES FIELD 1!)
        f1_scalar.chunk.emit(Instruction::NewStruct(1, pair_name1s, a_name1s, 1, 2), 2);
        // IP 7: AddInt R3 = R3 + R4
        f1_scalar.chunk.emit(Instruction::AddInt(3, 3, 4), 2);
        // IP 8: Jump back to IP 3 (3 - 9 = -6)
        f1_scalar.chunk.emit(Instruction::Jump(-6), 2);
        // IP 9: Return R1
        f1_scalar.chunk.emit(Instruction::Return(1), 3);

        // Case 2: Loop trace with NewStruct destination aliasing second input (R2)
        // and mid-trace bailout (bailout guard when i reaches 25).
        // Proves:
        // 1) Guard bailout exit via stats.guard_exits increment;
        // 2) Observable concrete state after bailout: loop counter R3 == 25 (not 50);
        // 3) Value of the overlapping second field (index 1) in the struct created right before bailout;
        // 4) Value of the non-overlapping first field (index 0) == 42.
        let mut f2 = Function::new("trace_loop_alias_second_bailout", 1, false);
        f2.set_register_count(10);

        let pair_name2 = f2.chunk.add_constant(Value::string("Pair"));
        let a_name2 = f2.chunk.add_constant(Value::string("a"));
        let _b_name2 = f2.chunk.add_constant(Value::string("b"));
        let val_42_2 = f2.chunk.add_constant(Value::Int(42));
        let val_99_2 = f2.chunk.add_constant(Value::Int(99));
        let val_0_2 = f2.chunk.add_constant(Value::Int(0));
        let val_1_2 = f2.chunk.add_constant(Value::Int(1));
        let val_25 = f2.chunk.add_constant(Value::Int(25));
        let val_999 = f2.chunk.add_constant(Value::Int(999));

        // IP 0: LoadConst R1 = 42
        f2.chunk.emit(Instruction::LoadConst(1, val_42_2), 1);
        // IP 1: LoadConst R2 = 99
        f2.chunk.emit(Instruction::LoadConst(2, val_99_2), 1);
        // IP 2: LoadConst R3 = 0
        f2.chunk.emit(Instruction::LoadConst(3, val_0_2), 1);
        // IP 3: LoadConst R4 = 1
        f2.chunk.emit(Instruction::LoadConst(4, val_1_2), 1);
        // IP 4: LoadConst R6 = 25
        f2.chunk.emit(Instruction::LoadConst(6, val_25), 1);

        // Loop header at IP 5:
        // IP 5: LtInt R5 = R3 < R0
        f2.chunk.emit(Instruction::LtInt(5, 3, 0), 2);
        // IP 6: JumpIfNot R5, +5 (to IP 12 Normal Exit: 7 + 5 = 12)
        f2.chunk.emit(Instruction::JumpIfNot(5, 5), 2);
        // IP 7: NewStruct dest=2, first_field_reg=1, count=2 (DEST 2 ALIASES FIELD 2!)
        f2.chunk.emit(Instruction::NewStruct(2, pair_name2, a_name2, 1, 2), 2);
        // IP 8: LtInt R7 = R3 < R6 (in-trace guard: when R3 reaches 25, bails out to IP 14)
        f2.chunk.emit(Instruction::LtInt(7, 3, 6), 2);
        // IP 9: JumpIfNot R7, +4 (to IP 14 Bailout Handler: 10 + 4 = 14)
        f2.chunk.emit(Instruction::JumpIfNot(7, 4), 2);
        // IP 10: AddInt R3 = R3 + R4
        f2.chunk.emit(Instruction::AddInt(3, 3, 4), 2);
        // IP 11: Jump back to IP 5 (offset: 5 - 12 = -7)
        f2.chunk.emit(Instruction::Jump(-7), 2);

        // Normal exit at IP 12 (returned if loop ran to limit without bailout):
        // IP 12: LoadConst R9 = 999
        f2.chunk.emit(Instruction::LoadConst(9, val_999), 3);
        // IP 13: Return R9
        f2.chunk.emit(Instruction::Return(9), 3);

        // Bailout handler at IP 14 (entered when guard fails in JIT):
        // IP 14: Move R4 = R2 (move R2 struct into R4; R3 holds 25, R4 holds struct)
        f2.chunk.emit(Instruction::Move(4, 2), 3);
        // IP 15: NewStruct dest=8, [R3, R4] -> Pair { a: 25, b: R2 }
        f2.chunk.emit(Instruction::NewStruct(8, pair_name2, a_name2, 3, 2), 3);
        // IP 16: Return R8
        f2.chunk.emit(Instruction::Return(8), 3);

        // Case 2b: Loop trace with NewStruct dest=2 aliasing second field reg=2,
        // with scalar update each iteration: R1 = 42, R2 = R3.
        // Directly verifies the scalar value of the overlapping field (field 1) at loop exit.
        let mut f2_scalar = Function::new("trace_loop_alias_second_scalar", 1, false);
        f2_scalar.set_register_count(6);
        let pair_name2s = f2_scalar.chunk.add_constant(Value::string("Pair"));
        let a_name2s = f2_scalar.chunk.add_constant(Value::string("a"));
        let _b_name2s = f2_scalar.chunk.add_constant(Value::string("b"));
        let val_42_2s = f2_scalar.chunk.add_constant(Value::Int(42));
        let val_0_2s = f2_scalar.chunk.add_constant(Value::Int(0));
        let val_1_2s = f2_scalar.chunk.add_constant(Value::Int(1));

        // IP 0: LoadConst R1 = 42
        f2_scalar.chunk.emit(Instruction::LoadConst(1, val_42_2s), 1);
        // IP 1: LoadConst R3 = 0
        f2_scalar.chunk.emit(Instruction::LoadConst(3, val_0_2s), 1);
        // IP 2: LoadConst R4 = 1
        f2_scalar.chunk.emit(Instruction::LoadConst(4, val_1_2s), 1);

        // Loop header at IP 3:
        // IP 3: LtInt R5 = R3 < R0
        f2_scalar.chunk.emit(Instruction::LtInt(5, 3, 0), 2);
        // IP 4: JumpIfNot R5, +4 (to IP 9 Return R2: 5 + 4 = 9)
        f2_scalar.chunk.emit(Instruction::JumpIfNot(5, 4), 2);
        // IP 5: Move R2 = R3 (current counter value into R2)
        f2_scalar.chunk.emit(Instruction::Move(2, 3), 2);
        // IP 6: NewStruct dest=2, first_field_reg=1, count=2 (DEST 2 ALIASES FIELD 2!)
        f2_scalar.chunk.emit(Instruction::NewStruct(2, pair_name2s, a_name2s, 1, 2), 2);
        // IP 7: AddInt R3 = R3 + R4
        f2_scalar.chunk.emit(Instruction::AddInt(3, 3, 4), 2);
        // IP 8: Jump back to IP 3 (3 - 9 = -6)
        f2_scalar.chunk.emit(Instruction::Jump(-6), 2);
        // IP 9: Return R2
        f2_scalar.chunk.emit(Instruction::Return(2), 3);

        vm.load_functions(vec![f1, f1_scalar, f2, f2_scalar]);

        // Run Case 1: Trace loop with aliased first field (recursive chaining)
        let res1 = vm.call("trace_loop_alias_first", vec![Value::Int(50)]).unwrap();
        if let Value::Struct(obj) = &res1 {
            let fields = obj.fields.borrow();
            assert_eq!(fields[1], Value::Int(99), "non-overlapping field b must be 99");

            // Verify the OVERLAPPING field a (index 0):
            // Every iteration wrapped the previous R1 into field 0.
            // Walk down all 50 levels of nesting down to the initial Value::Int(42).
            let mut depth = 0;
            let mut curr = res1.clone();
            while let Value::Struct(curr_obj) = curr {
                depth += 1;
                let curr_fields = curr_obj.fields.borrow();
                assert_eq!(curr_fields[1], Value::Int(99), "field b at depth {depth} must be 99");
                curr = curr_fields[0].clone();
            }
            assert_eq!(depth, 50, "must traverse exactly 50 iterations of nested struct creation");
            assert_eq!(curr, Value::Int(42), "innermost overlapping field a must be initial value 42");
        } else {
            panic!("expected Struct, got {:?}", res1);
        }

        let stats1 = vm.jit_stats();
        assert!(
            stats1.root_traces_compiled >= 1,
            "TraceRecorder must successfully compile a loop trace when NewStruct dest aliases an input"
        );
        assert!(
            stats1.native_trace_entries > 0,
            "native trace execution must be entered"
        );
        assert!(
            stats1.struct_fast_path_hits > 0,
            "fast path must be hit in native trace"
        );

        // Run Case 1b: Trace loop with aliased first field with scalar reload
        let res1_scalar = vm.call("trace_loop_alias_first_scalar", vec![Value::Int(50)]).unwrap();
        if let Value::Struct(obj) = res1_scalar {
            let fields = obj.fields.borrow();
            // Field 0 is the OVERLAPPING field, set from counter R3 on last iteration (49)
            assert_eq!(
                fields[0],
                Value::Int(49),
                "overlapping field a must receive the exact scalar value 49 on last iteration"
            );
            assert_eq!(fields[1], Value::Int(99), "non-overlapping field b must be 99");
        } else {
            panic!("expected Struct, got {:?}", res1_scalar);
        }

        let prev_traces = vm.jit_stats().root_traces_compiled;
        let prev_native = vm.jit_stats().native_trace_entries;
        let prev_guards = vm.jit_stats().guard_exits;

        // Run Case 2: Trace loop with aliased second field and mid-trace bailout
        let res2 = vm.call("trace_loop_alias_second_bailout", vec![Value::Int(50)]).unwrap();
        if let Value::Struct(obj) = &res2 {
            let fields = obj.fields.borrow();
            // Proves bailout: R3 (counter) was 25 at guard exit, packaged into Pair.a
            assert_eq!(
                fields[0],
                Value::Int(25),
                "bailout handler must receive counter R3 = 25 (proves bailout occurred at guard, not at loop limit 50)"
            );

            // fields[1] is R2 (the struct created right before the guard)
            if let Value::Struct(inner_obj) = &fields[1] {
                let inner_fields = inner_obj.fields.borrow();
                assert_eq!(
                    inner_fields[0],
                    Value::Int(42),
                    "non-overlapping field a must be 42"
                );

                // Verify the OVERLAPPING field b (index 1):
                // Traverses the 26 levels of nesting (iterations 0..=25) down to initial 99
                let mut depth = 0;
                let mut curr = fields[1].clone();
                while let Value::Struct(curr_obj) = curr {
                    depth += 1;
                    let curr_fields = curr_obj.fields.borrow();
                    assert_eq!(curr_fields[0], Value::Int(42), "field a at depth {depth} must be 42");
                    curr = curr_fields[1].clone();
                }
                assert_eq!(depth, 26, "must traverse exactly 26 iterations up to the guard failure point");
                assert_eq!(curr, Value::Int(99), "innermost overlapping field b must be initial value 99");
            } else {
                panic!("expected Struct R2 in bailout Pair.b");
            }
        } else {
            panic!("expected Struct, got {:?}", res2);
        }

        let stats2 = vm.jit_stats();
        assert!(
            stats2.root_traces_compiled > prev_traces,
            "second loop trace must also compile"
        );
        assert!(
            stats2.native_trace_entries > prev_native,
            "second native trace execution must be entered"
        );
        assert_eq!(
            stats2.guard_exits - prev_guards,
            1,
            "exactly one guard exit must be recorded from the mid-trace bailout"
        );

        // Run Case 2b: Trace loop with aliased second field with scalar reload
        let res2_scalar = vm.call("trace_loop_alias_second_scalar", vec![Value::Int(50)]).unwrap();
        if let Value::Struct(obj) = res2_scalar {
            let fields = obj.fields.borrow();
            assert_eq!(fields[0], Value::Int(42), "non-overlapping field a must be 42");
            // Field 1 is the OVERLAPPING field, set from counter R3 on last iteration (49)
            assert_eq!(
                fields[1],
                Value::Int(49),
                "overlapping field b must receive the exact scalar value 49 on last iteration"
            );
        } else {
            panic!("expected Struct, got {:?}", res2_scalar);
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn jit_struct_cyclic_gc_live_weak_upgrade_real_jit() {
        use crate::ast::{Span, Type, TypeKind};
        use crate::bytecode::value::{FieldStorage, StructLayout};
        use std::rc::Rc;

        let mut vm = VM::new();
        let any_ty = Type::new(TypeKind::Unknown, Span::dummy());
        let layout = Rc::new(StructLayout::new(
            "CycleNode".to_string(),
            vec![Rc::new("val".to_string()), Rc::new("partner".to_string())],
            vec![FieldStorage::Strong, FieldStorage::Strong],
            vec![any_ty.clone(), any_ty.clone()],
            vec![None, None],
        ));
        vm.struct_metadata.insert(
            "CycleNode".to_string(),
            crate::vm::RuntimeStructInfo { layout: layout.clone() },
        );

        // Function make_cyclic_pair: runs a hot loop in JIT, creating mutually referencing CycleNodes:
        // NodeA.partner = NodeB, NodeB.partner = NodeA.
        // It returns the final NodeA.
        let mut f = Function::new("make_cyclic_pair", 1, false);
        f.set_register_count(11);

        let node_name = f.chunk.add_constant(Value::string("CycleNode"));
        let val_name = f.chunk.add_constant(Value::string("val"));
        let _partner_name = f.chunk.add_constant(Value::string("partner"));
        let val_1 = f.chunk.add_constant(Value::Int(1));
        let val_2 = f.chunk.add_constant(Value::Int(2));
        let val_0 = f.chunk.add_constant(Value::Int(0));
        let val_step = f.chunk.add_constant(Value::Int(1));

        // R0: loop limit (30)
        // R1: loop counter 'i' (0)
        // R2: step (1)
        // R3: Nil
        // R4: val 1
        // R5: Nil (slot for NodeA partner initial)
        // R6: NodeA
        // R7: NodeB
        // R8: val 2
        // R9: NodeA (slot for NodeB partner)
        // R10: condition (i < limit)
        f.chunk.emit(Instruction::LoadConst(1, val_0), 1);
        f.chunk.emit(Instruction::LoadConst(2, val_step), 1);
        f.chunk.emit(Instruction::LoadConst(4, val_1), 1);
        f.chunk.emit(Instruction::LoadNil(5), 1);
        f.chunk.emit(Instruction::LoadConst(8, val_2), 1);

        // Loop header at IP 5:
        // IP 5: LtInt R10 = R1 < R0
        f.chunk.emit(Instruction::LtInt(10, 1, 0), 2);
        // IP 6: JumpIfNot R10, +6 (to IP 13 Return R6: 7 + 6 = 13)
        f.chunk.emit(Instruction::JumpIfNot(10, 6), 2);
        // IP 7: NewStruct dest=6, [R4, R5] (NodeA: val=1, partner=Nil)
        f.chunk.emit(Instruction::NewStruct(6, node_name, val_name, 4, 2), 2);
        // IP 8: Move R9 = R6 (NodeA into R9)
        f.chunk.emit(Instruction::Move(9, 6), 2);
        // IP 9: NewStruct dest=7, [R8, R9] (NodeB: val=2, partner=NodeA)
        f.chunk.emit(Instruction::NewStruct(7, node_name, val_name, 8, 2), 2);
        // IP 10: SetField object=6, "partner", value=7 (NodeA.partner = NodeB)
        f.chunk.emit(Instruction::SetField(6, _partner_name, 7), 2);
        // IP 11: AddInt R1 = R1 + R2
        f.chunk.emit(Instruction::AddInt(1, 1, 2), 2);
        // IP 12: Jump back to IP 5 (offset: 5 - 13 = -8)
        f.chunk.emit(Instruction::Jump(-8), 2);

        // IP 13: Return R6
        f.chunk.emit(Instruction::Return(6), 3);

        vm.load_functions(vec![f]);

        let res = vm.call("make_cyclic_pair", vec![Value::Int(30)]).unwrap();
        let stats = vm.jit_stats();
        assert!(stats.root_traces_compiled >= 1, "cyclic loop trace must compile in JIT");
        assert!(stats.native_trace_entries > 0, "native JIT trace must execute");
        assert!(stats.struct_fast_path_hits > 0, "fast path hits must be recorded");

        let (weak_a, weak_b) = if let Value::Struct(ref node_a) = res {
            let weak_a = Rc::downgrade(node_a);
            let fields_a = node_a.fields.borrow();
            let weak_b = if let Value::Struct(ref node_b) = fields_a[1] {
                let fields_b = node_b.fields.borrow();
                // Verify mutual circular reference: NodeB.partner points back to NodeA!
                if let Value::Struct(ref back_to_a) = fields_b[1] {
                    assert!(
                        Rc::ptr_eq(node_a, back_to_a),
                        "NodeB.partner must point directly back to NodeA (closed mutual cycle)"
                    );
                } else {
                    panic!("expected NodeB.partner to point back to NodeA");
                }
                Rc::downgrade(node_b)
            } else {
                panic!("expected NodeB in NodeA.partner");
            };
            (weak_a, weak_b)
        } else {
            panic!("expected Struct NodeA");
        };

        // 1. Live Weak handles before dropping root: both upgraded successfully
        assert!(weak_a.upgrade().is_some(), "NodeA must be alive");
        assert!(weak_b.upgrade().is_some(), "NodeB must be alive");

        // 2. Drop the returned root handle from Rust scope.
        // At this point, NO external references to NodeA or NodeB exist in Rust or VM roots.
        drop(res);

        // 3. Both nodes MUST STILL BE ALIVE because of the mutual reference cycle!
        // Standard Rust Rc reference counting CANNOT reclaim them (strong_count == 1 for each).
        // This proves that we have a genuine uncollected cyclic structure in memory.
        assert!(
            weak_a.upgrade().is_some(),
            "NodeA must remain alive in memory due to cyclic reference holding it"
        );
        assert!(
            weak_b.upgrade().is_some(),
            "NodeB must remain alive in memory due to cyclic reference holding it"
        );
        // Note: upgrade() temporarily increments strong_count by 1 for the upgraded Rc handle itself.
        // Therefore, exactly 1 reference comes from the partner's circular reference field.
        assert_eq!(Rc::strong_count(&weak_a.upgrade().unwrap()), 2, "1 from cycle edge + 1 from temporary upgrade");
        assert_eq!(Rc::strong_count(&weak_b.upgrade().unwrap()), 2, "1 from cycle edge + 1 from temporary upgrade");

        // 4. Force cycle collection: the cycle collector discovers the unreachable cyclic cluster
        // and breaks the cycle by zeroing fields.
        vm.force_collect_cycles();

        // 5. Irrefutable proof of actual deallocation:
        // Live Weak handles must now fail to upgrade (strong_count == 0)!
        assert!(
            weak_a.upgrade().is_none(),
            "NodeA must be completely deallocated by cycle collector (weak handle cannot upgrade)"
        );
        assert!(
            weak_b.upgrade().is_none(),
            "NodeB must be completely deallocated by cycle collector (weak handle cannot upgrade)"
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn bool_array_vm_jit_differential() {
        use crate::config::LustConfig;

        // An algorithmic workload on boolean arrays:
        // Populates an array of 500 booleans (true),
        // clears every 3rd index to false in a loop,
        // then counts the remaining true values.
        let source = r#"
            function sieve_sub(n: int): int
                local flags: Array<bool> = []
                local i: int = 0
                while i < n do
                    array.push(flags, true)
                    i = i + 1
                end
                i = 0
                while i < n do
                    if i % 3 == 0 then
                        flags[i] = false
                    end
                    i = i + 1
                end
                local count: int = 0
                i = 0
                while i < n do
                    if flags[i] is Ok(f) then
                        if f then
                            count = count + 1
                        end
                    end
                    i = i + 1
                end
                return count
            end
        "#;

        // Run with VM (interpreter, JIT disabled)
        let mut vm_cfg = LustConfig::default();
        vm_cfg.set_jit_enabled(false);
        let mut vm_prog = crate::EmbeddedProgram::builder()
            .with_config(vm_cfg)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();
        let vm_res = vm_prog.call_raw("main.sieve_sub", vec![Value::Int(500)]).unwrap();

        // Run with JIT (JIT enabled, hot trace executed)
        let mut jit_cfg = LustConfig::default();
        jit_cfg.set_jit_enabled(true);
        let mut jit_prog = crate::EmbeddedProgram::builder()
            .with_config(jit_cfg)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();
        let jit_res = jit_prog.call_raw("main.sieve_sub", vec![Value::Int(500)]).unwrap();

        let stats = jit_prog.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "JIT must compile root traces for hot loop");
        assert!(stats.native_trace_entries > 0, "JIT must execute native traces");

        // Verify differential: VM and JIT produce identical count!
        assert_eq!(vm_res, jit_res, "VM and JIT results must be strictly identical");
        assert_eq!(vm_res, Value::Int(333), "500 - (500/3 + 1) = 333 true values");
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn bool_array_mixed_values_guard_bailout() {
        use crate::config::LustConfig;

        // Function takes an array of booleans, hot-loops reading it,
        // with a mixed integer injected at index 80.
        // Loop becomes hot (threshold 50), records and compiles for booleans,
        // enters native trace, and bails out via guard exit at index 80.
        let source = r#"
            function sum_booleans(arr: Array<unknown>, limit: int): int
                local count: int = 0
                local i: int = 0
                while i < limit do
                    if arr[i] is Ok(v) then
                        if v == true then
                            count = count + 1
                        end
                    end
                    i = i + 1
                end
                return count
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        // Build array with 120 elements: indices 0..79 and 81..119 are booleans, index 80 is Int(99)
        let mut values: Vec<Value> = (0..120).map(|_| Value::Bool(true)).collect();
        values[80] = Value::Int(99); // mixed value encountered after native JIT entry
        let arr_val = Value::array(values);

        let res = program.call_raw("main.sum_booleans", vec![arr_val.clone(), Value::Int(120)]).unwrap();
        // 119 elements are true, 1 is 99 (not true)
        assert_eq!(res, Value::Int(119));

        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "trace must compile for hot boolean loop");
        assert!(stats.native_trace_entries > 0, "native trace must be entered");
        assert!(stats.guard_exits > 0, "mixed value must trigger guard exit to interpreter");

        // Verify the array preserves both booleans and the mixed integer
        if let Value::Array(ref rc) = arr_val {
            let borrowed = rc.borrow();
            assert_eq!(borrowed[0], Value::Bool(true));
            assert_eq!(borrowed[80], Value::Int(99));
            assert_eq!(borrowed[119], Value::Bool(true));
        } else {
            panic!("expected Array");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn bool_array_shared_aliases_propagation() {
        use crate::config::LustConfig;

        // Proves that aliases 'a' and 'b' share identity:
        // writes to 'b' in hot JIT loop mutate 'a' directly.
        let source = r#"
            function alias_mutation(limit: int): Array<bool>
                local a: Array<bool> = []
                local i: int = 0
                while i < limit do
                    array.push(a, true)
                    i = i + 1
                end
                local b: Array<bool> = a
                i = 0
                while i < limit do
                    b[i] = false
                    i = i + 1
                end
                return a
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let res = program.call_raw("main.alias_mutation", vec![Value::Int(80)]).unwrap();
        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1);
        assert!(stats.native_trace_entries > 0);

        // Every element of 'a' must have been updated to false via 'b'
        if let Value::Array(ref rc) = res {
            let borrowed = rc.borrow();
            assert_eq!(borrowed.len(), 80);
            assert!(borrowed.iter().all(|v| *v == Value::Bool(false)));
        } else {
            panic!("expected Array");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn bool_array_ffi_borrow_and_mutations() {
        use crate::config::LustConfig;
        use crate::embed::{ArrayHandle, FromLustValue};

        let source = r#"
            function make_flags(): Array<bool>
                local arr: Array<bool> = []
                array.push(arr, true)
                array.push(arr, false)
                array.push(arr, true)
                return arr
            end

            function read_first_two(arr: Array<unknown>): int
                local c: int = 0
                if arr[0] is Ok(v0) then
                    if v0 == true then c = c + 1 end
                end
                if arr[1] is Ok(v1) then
                    if v1 == true then c = c + 1 end
                end
                return c
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let arr_val = program.call_raw("main.make_flags", vec![]).unwrap();
        let handle = ArrayHandle::from_value(arr_val.clone()).expect("must convert to ArrayHandle");

        let is_uniform_bool = |h: &ArrayHandle| {
            let s = h.borrow();
            !s.is_empty() && s.iter().all(|v| matches!(v, Value::Bool(_)))
        };

        // 1. FFI Public borrow() returning Ref<'_, [Value]>
        {
            let slice: core::cell::Ref<'_, [Value]> = handle.borrow();
            assert_eq!(slice.len(), 3);
            assert_eq!(slice[0], Value::Bool(true));
            assert_eq!(slice[1], Value::Bool(false));
            assert_eq!(slice[2], Value::Bool(true));
            assert!(is_uniform_bool(&handle));
        }

        // 2. FFI Public borrow_mut() returning RefMut<'_, Vec<Value>>
        {
            let mut vec_mut = handle.borrow_mut();
            // Mutate in-place
            vec_mut[1] = Value::Bool(true); // now [true, true, true]
            // Push a mixed value
            vec_mut.push(Value::Int(42));
        }

        assert!(!is_uniform_bool(&handle), "array now has mixed integer");
        assert_eq!(handle.len(), 4);

        // 3. Lust function execution reflects the FFI mutations
        let count = program.call_raw("main.read_first_two", vec![arr_val.clone()]).unwrap();
        assert_eq!(count, Value::Int(2), "arr[0] and arr[1] are now both true");
    }

    #[test]
    fn bool_array_budget_failure_atomicity() {
        use crate::EmbeddedProgram;
        use crate::embed::{ArrayHandle, FromLustValue};

        let source = r#"
            function grow_flags(arr: Array<bool>): ()
                while true do
                    array.push(arr, true)
                end
            end
        "#;

        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let arr_val = Value::array(vec![Value::Bool(true), Value::Bool(false)]);
        let handle = ArrayHandle::from_value(arr_val.clone()).unwrap();

        // Set a small memory budget (e.g. 500 bytes)
        program.vm_mut().set_memory_budget_bytes(500).unwrap();

        let err = program.call_raw("main.grow_flags", vec![arr_val.clone()]).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.to_lowercase().contains("memory budget"), "must trap with memory budget: {msg}");

        // Budget Failure Atomicity:
        // The array state must be valid and intact; previous elements are preserved.
        let slice = handle.borrow();
        assert!(slice.len() >= 2);
        assert_eq!(slice[0], Value::Bool(true));
        assert_eq!(slice[1], Value::Bool(false));
        for v in slice.iter() {
            assert!(matches!(v, Value::Bool(_)));
        }
    }

    #[test]
    fn bool_array_cyclic_gc_tracking() {
        use crate::EmbeddedProgram;
        use std::rc::Rc;

        let source = r#"
            struct Node
                flags: Array<unknown>
            end

            function make_cyclic_graph(): Array<unknown>
                local arr: Array<unknown> = [true, false]
                local node = Node { flags = arr }
                array.push(arr, node)
                return arr
            end
        "#;

        let mut program = EmbeddedProgram::builder()
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let arr_val = program.call_raw("main.make_cyclic_graph", vec![]).unwrap();
        let weak_arr = if let Value::Array(ref rc) = arr_val {
            Rc::downgrade(rc)
        } else {
            panic!("expected Array");
        };

        assert!(weak_arr.upgrade().is_some());
        // Verify cycle collector tracks array containers
        assert!(program.vm().cycle_collector.tracked_count() > 0);

        // Drop root in Rust
        drop(arr_val);

        // Still alive because arr contains node, and node.flags points back to arr (cycle)
        assert!(weak_arr.upgrade().is_some(), "array must remain alive in memory due to cyclic reference");

        // Force cycle collection
        program.vm_mut().force_collect_cycles();

        // Container deallocation verified via Weak reference after cycle collection
        assert!(
            weak_arr.upgrade().is_none(),
            "cyclic array must be deallocated by cycle collector"
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn bool_array_resize_realloc_hot_trace() {
        use crate::config::LustConfig;

        // Hot loop pushing 3000 booleans into an empty array.
        // Exercises geometric growth through capacity reallocations in compiled trace.
        let source = r#"
            function push_many(count: int): Array<bool>
                local arr: Array<bool> = []
                local i: int = 0
                while i < count do
                    array.push(arr, i % 2 == 0)
                    i = i + 1
                end
                return arr
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let res = program.call_raw("main.push_many", vec![Value::Int(3000)]).unwrap();
        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1);
        assert!(stats.native_trace_entries > 0);

        if let Value::Array(ref rc) = res {
            let borrowed = rc.borrow();
            assert_eq!(borrowed.len(), 3000);
            for (idx, val) in borrowed.iter().enumerate() {
                assert_eq!(*val, Value::Bool(idx % 2 == 0));
            }
        } else {
            panic!("expected Array");
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn sieve_push_loop_trace_proof_and_safety() {
        use crate::config::LustConfig;

        // 1. Proof of compiled trace with native entry for sieve array initialization
        let source = r#"
            function init_sieve(n: int): Array<bool>
                local flags: Array<bool> = []
                local i: int = 0
                while i <= n do
                    array.push(flags, true)
                    i = i + 1
                end
                return flags
            end

            function alias_push(arr: Array<bool>, count: int): ()
                local alias = arr
                local i: int = 0
                while i < count do
                    array.push(alias, false)
                    i = i + 1
                end
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        // Push across multiple geometric growth boundaries (cap: 0 -> 4 -> 8 -> ... -> 16384)
        let res = program.call_raw("main.init_sieve", vec![Value::Int(10000)]).unwrap();
        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "sieve push loop must compile root trace");
        assert!(stats.native_trace_entries > 0, "sieve push loop must execute natively");

        if let Value::Array(ref rc) = res {
            assert_eq!(rc.borrow().len(), 10001);
            assert!(rc.borrow().iter().all(|v| *v == Value::Bool(true)));
        } else {
            panic!("expected Array");
        }

        // 2. Alias safety: push through alias mutates original array across reallocations
        program.call_raw("main.alias_push", vec![res.clone(), Value::Int(500)]).unwrap();
        if let Value::Array(ref rc) = res {
            assert_eq!(rc.borrow().len(), 10501);
            assert_eq!(rc.borrow()[10000], Value::Bool(true));
            assert_eq!(rc.borrow()[10001], Value::Bool(false));
            assert_eq!(rc.borrow()[10500], Value::Bool(false));
        }

        // 3. Memory budget enforcement on growth: traps cleanly preserving array contents
        program.vm_mut().set_memory_budget_bytes(500).unwrap();
        let err = program.call_raw("main.init_sieve", vec![Value::Int(1000)]).unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("memory budget"));
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn sieve_002_callable_substitution_defect_red() {
        use crate::config::LustConfig;

        let source = r#"
            function dummy_pusher(arr: Array<bool>, val: bool): ()
                -- intentionally do NOT push to arr
            end

            function loop_caller(p: function(Array<bool>, bool): (), arr: Array<bool>, n: int): ()
                local i = 0
                while i < n do
                    p(arr, true)
                    i = i + 1
                end
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        // Get array.push native function
        let array_mod = program.vm().get_global("array").expect("array module");
        let array_push = if let Value::Map(ref m) = array_mod {
            m.borrow().get(&crate::bytecode::ValueKey::from("push")).cloned().unwrap()
        } else {
            panic!("expected array Map");
        };

        // 1. Train JIT trace with array.push
        let arr1 = Value::array(Vec::new());
        program
            .call_raw("main.loop_caller", vec![array_push.clone(), arr1.clone(), Value::Int(1000)])
            .unwrap();

        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "loop must compile a root trace");
        assert!(stats.native_trace_entries > 0, "loop must execute native trace");
        if let Value::Array(ref rc) = arr1 {
            assert_eq!(rc.borrow().len(), 1000, "arr1 must have 1000 pushed items");
        }

        // Create host callback dummy_fn
        let dummy_invocations = std::rc::Rc::new(std::cell::Cell::new(0usize));
        let dummy_invocations_clone = dummy_invocations.clone();
        let dummy_fn = Value::NativeFunction(crate::bytecode::value::native_fn(move |_args| {
            dummy_invocations_clone.set(dummy_invocations_clone.get() + 1);
            Ok(NativeCallResult::Return(Value::Nil))
        }));

        // 2. Now call loop_caller with dummy_fn!
        // In VM / semantics: dummy_pusher does NOT push anything to arr2, so arr2 length should remain 0!
        let arr2 = Value::array(Vec::new());
        let prev_entries = program.vm().jit_stats().native_trace_entries;
        program
            .call_raw("main.loop_caller", vec![dummy_fn, arr2.clone(), Value::Int(1000)])
            .unwrap();


        let post_stats = program.vm().jit_stats();
        assert!(post_stats.native_trace_entries > prev_entries, "must have entered native trace");

        if let Value::Array(ref rc) = arr2 {
            // If defective (missing GuardNativeFunction), the JIT blindly ran ArrayPush and len == 1000!
            // Correct semantics: dummy_pusher was called, len == 0!
            assert_eq!(
                rc.borrow().len(),
                0,
                "callable was substituted with dummy_pusher; arr2 must NOT have items pushed! (Defect: JIT ran ArrayPush without guarding callable!)"
            );
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn sieve_002_guard_globals_mutable_map_red() {
        use crate::config::LustConfig;
        use std::cell::Cell;
        use std::rc::Rc;

        let source = r#"
            function loop_push(arr: Array<bool>, n: int): ()
                local i = 0
                while i < n do
                    array.push(arr, true)
                    i = i + 1
                end
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        // 1. Train JIT trace with default array.push
        let arr1 = Value::array(Vec::new());
        program
            .call_raw("main.loop_push", vec![arr1.clone(), Value::Int(1000)])
            .unwrap();

        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "must compile root trace");
        assert!(stats.native_trace_entries > 0, "must execute native trace");
        if let Value::Array(ref rc) = arr1 {
            assert_eq!(rc.borrow().len(), 1000);
        }

        // 2. Now replace array.push inside the array module map with a custom native function!
        let custom_calls = Rc::new(Cell::new(0usize));
        let custom_calls_clone = custom_calls.clone();
        let custom_push = Value::NativeFunction(crate::bytecode::value::native_fn(move |_args| {
            custom_calls_clone.set(custom_calls_clone.get() + 1);
            Ok(NativeCallResult::Return(Value::Nil))
        }));

        let array_mod = program.vm().get_global("array").expect("array module");
        let version_before = program.vm().globals_version;
        if let Value::Map(ref m) = array_mod {
            m.borrow_mut().insert(crate::bytecode::ValueKey::from("push"), custom_push.clone());
        } else {
            panic!("expected Map");
        }

        // Critical verification: globals_version was NOT changed because array is a mutable Map mutated in-place
        assert_eq!(
            program.vm().globals_version,
            version_before,
            "globals_version is NOT updated by map.borrow_mut().insert()"
        );

        // 3. Call loop_push again with arr2
        let arr2 = Value::array(Vec::new());
        let prev_entries = program.vm().jit_stats().native_trace_entries;
        program
            .call_raw("main.loop_push", vec![arr2.clone(), Value::Int(1000)])
            .unwrap();

        let post_stats = program.vm().jit_stats();
        assert!(post_stats.native_trace_entries > prev_entries, "must have entered native trace");

        // Correct language semantics (VM without JIT):
        // array.push was replaced with custom_push (which does NOT push to arr2, but increments custom_calls).
        // Therefore, arr2.len() must be 0 and custom_calls must be 1000!
        // But under SIEVE-001 defect:
        // JIT trace baked in LoadConst array.push / ArrayPush under GuardGlobals (which passes!),
        // so JIT pushed 1000 items and custom_calls is 0!
        assert_eq!(
            custom_calls.get(),
            1000,
            "array.push was replaced in mutable Map via FFI/alias; custom_push MUST be invoked! (Defect: JIT baked in old push!)"
        );
        if let Value::Array(ref rc) = arr2 {
            assert_eq!(
                rc.borrow().len(),
                0,
                "arr2 must NOT have items pushed when array.push was replaced with custom_push"
            );
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn test_array_len_jit_generic_array_basic() {
        use crate::config::LustConfig;

        let source = r#"
            function sum_len(arr: Array<int>, n: int): int
                local total = 0
                local i = 0
                while i < n do
                    total = total + array.len(arr)
                    i = i + 1
                end
                return total
            end
        "#;

        let mut config_jit = LustConfig::default();
        config_jit.set_jit_enabled(true);
        let mut prog_jit = crate::EmbeddedProgram::builder()
            .with_config(config_jit)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let mut config_vm = LustConfig::default();
        config_vm.set_jit_enabled(false);
        let mut prog_vm = crate::EmbeddedProgram::builder()
            .with_config(config_vm)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let arr = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4), Value::Int(5)]);
        let res_jit = prog_jit
            .call_raw("main.sum_len", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        let res_vm = prog_vm
            .call_raw("main.sum_len", vec![arr.clone(), Value::Int(1000)])
            .unwrap();

        assert_eq!(res_jit, Value::Int(5000));
        assert_eq!(res_jit, res_vm, "JIT and VM results must match exactly");

        let stats = prog_jit.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "must compile root trace for array.len");
        assert!(stats.native_trace_entries > 0, "must execute native trace");
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn test_array_len_jit_callable_substitution_guard() {
        use crate::config::LustConfig;
        use std::cell::Cell;
        use std::rc::Rc;

        let source = r#"
            function sum_len(arr: Array<int>, n: int): int
                local total = 0
                local i = 0
                while i < n do
                    total = total + array.len(arr)
                    i = i + 1
                end
                return total
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let arr = Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);

        // 1. Compile and run trace with intrinsic array.len
        let res1 = program
            .call_raw("main.sum_len", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        assert_eq!(res1, Value::Int(3000));
        let stats = program.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1);
        assert!(stats.native_trace_entries > 0);

        // 2. Substitute array.len with custom function returning 42
        let custom_calls = Rc::new(Cell::new(0usize));
        let custom_calls_clone = custom_calls.clone();
        let custom_len = Value::NativeFunction(crate::bytecode::value::native_fn(move |_args| {
            custom_calls_clone.set(custom_calls_clone.get() + 1);
            Ok(NativeCallResult::Return(Value::Int(42)))
        }));

        let array_mod = program.vm().get_global("array").expect("array module");
        if let Value::Map(ref m) = array_mod {
            m.borrow_mut().insert(crate::bytecode::ValueKey::from("len"), custom_len);
        } else {
            panic!("expected Map");
        }

        // 3. Call again: callable substitution MUST be respected via GuardNativeFunction bailout
        let res2 = program
            .call_raw("main.sum_len", vec![arr.clone(), Value::Int(1000)])
            .unwrap();

        assert_eq!(res2, Value::Int(42000), "must invoke substituted custom callable, not intrinsic");
        assert_eq!(custom_calls.get(), 1000, "custom callable must have been called 1000 times");
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn test_array_len_jit_differential_loop_parity() {
        use crate::config::LustConfig;

        let source = r#"
            function process_arrays(n: int): int
                local arr: Array<int> = []
                local i = 0
                while i < 100 do
                    array.push(arr, i * 2)
                    i = i + 1
                end

                local total = 0
                local j = 0
                while j < n do
                    local len = array.len(arr)
                    local idx = j % len
                    if arr[idx] is Ok(v) then
                        total = total + v + len
                    end
                    j = j + 1
                end
                return total
            end
        "#;

        let mut config_jit = LustConfig::default();
        config_jit.set_jit_enabled(true);
        let mut prog_jit = crate::EmbeddedProgram::builder()
            .with_config(config_jit)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let mut config_vm = LustConfig::default();
        config_vm.set_jit_enabled(false);
        let mut prog_vm = crate::EmbeddedProgram::builder()
            .with_config(config_vm)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let res_jit = prog_jit
            .call_raw("main.process_arrays", vec![Value::Int(2000)])
            .unwrap();
        let res_vm = prog_vm
            .call_raw("main.process_arrays", vec![Value::Int(2000)])
            .unwrap();

        assert_eq!(res_jit, res_vm, "differential parity between JIT and VM failed");

        let stats = prog_jit.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1);
        assert!(stats.native_trace_entries > 0);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn test_array_len_jit_growth_shrinkage_alias_after_warmup() {
        use crate::config::LustConfig;

        let source = r#"
            function measure_loop(arr: Array<int>, n: int): int
                local sum = 0
                local i = 0
                while i < n do
                    sum = sum + array.len(arr)
                    i = i + 1
                end
                return sum
            end
        "#;

        let mut config_jit = LustConfig::default();
        config_jit.set_jit_enabled(true);
        let mut prog_jit = crate::EmbeddedProgram::builder()
            .with_config(config_jit)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let mut config_vm = LustConfig::default();
        config_vm.set_jit_enabled(false);
        let mut prog_vm = crate::EmbeddedProgram::builder()
            .with_config(config_vm)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let arr = Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);
        let arr_alias = arr.clone();

        // 1. Warmup loop to compile root trace: length is 3 -> 3 * 1000 = 3000
        let res_jit1 = prog_jit
            .call_raw("main.measure_loop", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        let res_vm1 = prog_vm
            .call_raw("main.measure_loop", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        assert_eq!(res_jit1, Value::Int(3000));
        assert_eq!(res_jit1, res_vm1);

        let stats1 = prog_jit.vm().jit_stats();
        assert!(stats1.root_traces_compiled >= 1, "must compile trace");
        let entries1 = stats1.native_trace_entries;
        assert!(entries1 > 0, "must execute native trace");

        // 2. Growth via alias: push 4 elements into arr_alias -> length is now 7
        if let Value::Array(ref rc) = arr_alias {
            let mut borrowed = rc.borrow_mut();
            for val in [40, 50, 60, 70] {
                borrowed.push(Value::Int(val));
            }
        }
        assert_eq!(arr.array_len(), Some(7));

        let res_jit2 = prog_jit
            .call_raw("main.measure_loop", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        let res_vm2 = prog_vm
            .call_raw("main.measure_loop", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        assert_eq!(res_jit2, Value::Int(7000), "must dynamically reflect growth via alias");
        assert_eq!(res_jit2, res_vm2);

        let stats2 = prog_jit.vm().jit_stats();
        assert!(
            stats2.native_trace_entries > entries1,
            "must continue executing native trace on grown array"
        );
        let entries2 = stats2.native_trace_entries;

        // 3. Shrinkage via alias: pop 5 elements -> length is now 2
        if let Value::Array(ref rc) = arr_alias {
            let mut borrowed = rc.borrow_mut();
            for _ in 0..5 {
                borrowed.pop();
            }
        }
        assert_eq!(arr.array_len(), Some(2));

        let res_jit3 = prog_jit
            .call_raw("main.measure_loop", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        let res_vm3 = prog_vm
            .call_raw("main.measure_loop", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        assert_eq!(res_jit3, Value::Int(2000), "must dynamically reflect shrinkage via alias");
        assert_eq!(res_jit3, res_vm3);

        let stats3 = prog_jit.vm().jit_stats();
        assert!(
            stats3.native_trace_entries > entries2,
            "must continue executing native trace on shrunk array"
        );
        let entries3 = stats3.native_trace_entries;

        // 4. Shrinkage to empty array: pop remaining 2 elements -> length is now 0
        if let Value::Array(ref rc) = arr_alias {
            let mut borrowed = rc.borrow_mut();
            borrowed.clear();
        }
        assert_eq!(arr.array_len(), Some(0));

        let res_jit4 = prog_jit
            .call_raw("main.measure_loop", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        let res_vm4 = prog_vm
            .call_raw("main.measure_loop", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        assert_eq!(res_jit4, Value::Int(0), "empty array length must evaluate to 0");
        assert_eq!(res_jit4, res_vm4);

        let stats4 = prog_jit.vm().jit_stats();
        assert!(
            stats4.native_trace_entries > entries3,
            "must continue executing native trace on empty array"
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn test_array_len_jit_ffi_callable_mutation_in_loop() {
        use crate::config::LustConfig;
        use std::cell::RefCell;
        use std::rc::Rc;

        let source = r#"
            function dynamic_mutate_loop(arr: Array<int>, step_fn: function(Array<int>): (), n: int): int
                local sum = 0
                local i = 0
                while i < n do
                    step_fn(arr)
                    sum = sum + array.len(arr)
                    i = i + 1
                end
                return sum
            end
        "#;

        let mut config_jit = LustConfig::default();
        config_jit.set_jit_enabled(true);
        let mut prog_jit = crate::EmbeddedProgram::builder()
            .with_config(config_jit)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let mut config_vm = LustConfig::default();
        config_vm.set_jit_enabled(false);
        let mut prog_vm = crate::EmbeddedProgram::builder()
            .with_config(config_vm)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        // Native step callable that oscillates array size between 3 and 7 elements
        let make_step_fn = || {
            let step_count = Rc::new(RefCell::new(0usize));
            Value::NativeFunction(crate::bytecode::value::native_fn(move |args| {
                let mut count = step_count.borrow_mut();
                *count += 1;
                let arr_val = args.first().expect("array arg");
                if let Value::Array(rc) = arr_val {
                    let mut b = rc.borrow_mut();
                    if *count % 2 == 0 {
                        b.push(Value::Int(*count as i64));
                    } else if b.len() > 3 {
                        b.pop();
                    }
                }
                Ok(NativeCallResult::Return(Value::Nil))
            }))
        };

        let arr_jit = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        let arr_vm = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);

        let res_jit = prog_jit
            .call_raw(
                "main.dynamic_mutate_loop",
                vec![arr_jit.clone(), make_step_fn(), Value::Int(500)],
            )
            .unwrap();
        let res_vm = prog_vm
            .call_raw(
                "main.dynamic_mutate_loop",
                vec![arr_vm.clone(), make_step_fn(), Value::Int(500)],
            )
            .unwrap();

        assert_eq!(res_jit, res_vm, "JIT with in-loop FFI mutation must match VM interpreter");

        let stats = prog_jit.vm().jit_stats();
        assert!(stats.root_traces_compiled >= 1, "must compile trace");
        assert!(stats.native_trace_entries > 0, "must execute native trace");
        assert_eq!(
            arr_jit.array_len(),
            arr_vm.array_len(),
            "final array lengths must match"
        );
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn test_array_len_jit_invalid_args_borrow_semantics_guard_bailout() {
        use crate::bytecode::value::jit_array_len_safe;
        use crate::config::LustConfig;
        use std::cell::Cell;
        use std::rc::Rc;

        // 1. Invalid arguments and arity directly on stdlib array.len callable
        let array_mod = crate::EmbeddedProgram::builder()
            .module("main", "function dummy(): () end")
            .entry_module("main")
            .compile()
            .unwrap()
            .vm()
            .get_global("array")
            .expect("array module");

        let array_len_fn = match array_mod {
            Value::Map(ref m) => m
                .borrow()
                .get(&crate::bytecode::ValueKey::from("len"))
                .cloned()
                .expect("len function"),
            _ => panic!("expected Map"),
        };

        if let Value::NativeFunction(ref nf) = array_len_fn {
            // Arity 0 / missing arg -> defaults to Nil -> error
            let err_empty = (nf)(&[]).unwrap_err();
            assert!(
                err_empty.contains("array.len expects an array"),
                "empty args must error: {err_empty}"
            );

            // Invalid arg types: Int, Float, String, Nil
            for bad_val in [
                Value::Int(123),
                Value::Float(4.56),
                Value::string("text"),
                Value::Nil,
            ] {
                let err = (nf)(&[bad_val.clone()]).unwrap_err();
                assert!(
                    err.contains("array.len expects an array"),
                    "bad value {:?} must error: {}",
                    bad_val,
                    err
                );
            }

            // Valid array -> succeeds
            let good_arr = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
            let res = (nf)(&[good_arr]).unwrap();
            match res {
                NativeCallResult::Return(Value::Int(3)) => {}
                other => panic!("expected Return(Int(3)), got {:?}", other),
            }
        } else {
            panic!("expected NativeFunction");
        }

        // 2. Low-level borrow semantics and null/non-array checks via jit_array_len_safe
        let test_arr = Value::array(vec![Value::Int(10), Value::Int(20), Value::Int(30)]);
        if let Value::Array(ref rc) = test_arr {
            // Normal unborrowed state: length is 3
            let len_normal = unsafe { jit_array_len_safe(&test_arr as *const Value) };
            assert_eq!(len_normal, 3);

            // Active mutable borrow: RefCell::try_borrow fails -> jit_array_len_safe returns -1
            {
                let _active_mut = rc.borrow_mut();
                let len_borrowed = unsafe { jit_array_len_safe(&test_arr as *const Value) };
                assert_eq!(
                    len_borrowed, -1,
                    "active borrow_mut MUST cause jit_array_len_safe to return -1 for bailout"
                );
            }

            // After mutable borrow is dropped: length restored to 3
            let len_restored = unsafe { jit_array_len_safe(&test_arr as *const Value) };
            assert_eq!(len_restored, 3);

            // Null pointer returns -1
            let len_null = unsafe { jit_array_len_safe(core::ptr::null()) };
            assert_eq!(len_null, -1);

            // Non-array Value returns -1
            let not_arr = Value::Int(999);
            let len_non_array = unsafe { jit_array_len_safe(&not_arr as *const Value) };
            assert_eq!(len_non_array, -1);
        } else {
            panic!("expected Array");
        }

        // 3. JIT Trace Guard Bailout and effect verification
        let source = r#"
            function sum_len_bailout(arr: Array<int>, n: int): int
                local total = 0
                local i = 0
                while i < n do
                    total = total + array.len(arr)
                    i = i + 1
                end
                return total
            end
        "#;

        let mut config = LustConfig::default();
        config.set_jit_enabled(true);
        let mut program = crate::EmbeddedProgram::builder()
            .with_config(config)
            .module("main", source)
            .entry_module("main")
            .compile()
            .unwrap();

        let arr = Value::array(vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)]);

        // Step A: Warmup and compile loop trace with standard intrinsic array.len
        let res_warm = program
            .call_raw("main.sum_len_bailout", vec![arr.clone(), Value::Int(1000)])
            .unwrap();
        assert_eq!(res_warm, Value::Int(4000));

        let stats_warm = program.vm().jit_stats();
        assert!(stats_warm.root_traces_compiled >= 1, "must compile root trace");
        assert!(stats_warm.native_trace_entries > 0, "must enter native trace");
        let guard_exits_before = stats_warm.guard_exits;

        // Step B: In-place substitution of array.len with custom callable
        let custom_invocations = Rc::new(Cell::new(0usize));
        let custom_invocations_clone = custom_invocations.clone();
        let custom_fn = Value::NativeFunction(crate::bytecode::value::native_fn(move |_args| {
            custom_invocations_clone.set(custom_invocations_clone.get() + 1);
            Ok(NativeCallResult::Return(Value::Int(99)))
        }));

        let array_global = program.vm().get_global("array").expect("array global");
        if let Value::Map(ref m) = array_global {
            m.borrow_mut().insert(crate::bytecode::ValueKey::from("len"), custom_fn);
        } else {
            panic!("expected Map");
        }

        // Step C: Run again. The compiled trace checks GuardNativeFunction on array.len.
        // It MUST fail the guard check, trigger a guard bailout, and fall back to interpreter.
        let res_after_bailout = program
            .call_raw("main.sum_len_bailout", vec![arr.clone(), Value::Int(1000)])
            .unwrap();

        assert_eq!(
            res_after_bailout,
            Value::Int(99000),
            "custom callable returning 99 must be executed after guard bailout (99 * 1000)"
        );
        assert_eq!(
            custom_invocations.get(),
            1000,
            "custom callable must have been called 1000 times"
        );

        let stats_after = program.vm().jit_stats();
        assert!(
            stats_after.guard_exits > guard_exits_before,
            "callable substitution MUST increment guard_exits (before: {}, after: {})",
            guard_exits_before,
            stats_after.guard_exits
        );
    }
}
