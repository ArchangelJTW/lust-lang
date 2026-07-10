use super::*;
use crate::ast::Type;
use crate::bytecode::{LustMap, ValueKey};
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
impl VM {
    pub fn new() -> Self {
        Self::with_config(&LustConfig::default())
    }

    pub fn with_config(config: &LustConfig) -> Self {
        let mut vm = Self {
            jit: JitState::new(),
            budgets: BudgetState::default(),
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
            #[cfg(feature = "std")]
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

    pub(crate) fn observe_value(&mut self, value: &Value) {
        self.cycle_collector.register_value(value);
    }

    pub(crate) fn observe_value_graph(&mut self, value: &Value) {
        self.cycle_collector.register_graph(value);
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
        for function in &functions {
            for value in &function.chunk.constants {
                self.observe_value_graph(value);
            }
        }
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
        self.observe_value_graph(&value);
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
        self.push_export_metadata(export);
        let native = Value::NativeFunction(Rc::new(move |args| {
            if args.len() != params.len() {
                return Err(format!(
                    "Native expects {} arguments, got {}",
                    params.len(),
                    args.len()
                ));
            }
            VM::with_current(|vm| {
                for (index, (value, param)) in args.iter().zip(&params).enumerate() {
                    if !vm.value_is_type(value, param.ty()) {
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
                    if !vm.value_is_type(value, expected) {
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
        let frame = self.make_call_frame(func_idx, None, args, Vec::new())?;

        let stack_depth_before = self.call_stack.len();
        let saved_pending_return_value = self.pending_return_value.clone();
        let saved_pending_return_dest = self.pending_return_dest;
        let saved_pending_task_signal = self.pending_task_signal.clone();
        let saved_last_task_signal = self.last_task_signal.clone();
        let saved_trace_recorder = self.trace_recorder.take();
        let saved_side_trace_context = self.side_trace_context.take();
        let saved_skip_next_trace_record = self.skip_next_trace_record;
        let saved_call_until_depth = self.call_until_depth;
        self.skip_next_trace_record = false;
        self.call_until_depth = Some(stack_depth_before);
        self.call_stack.push(frame);
        let result = self.run();
        self.trace_recorder = saved_trace_recorder;
        self.side_trace_context = saved_side_trace_context;
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
        vm.side_trace_context = Some((crate::jit::TraceId(4), 2));
        vm.skip_next_trace_record = true;

        assert!(vm.call("failing", Vec::new()).is_err());
        assert!(vm.call_stack.is_empty());
        assert!(vm.trace_recorder.is_some());
        assert_eq!(vm.side_trace_context, Some((crate::jit::TraceId(4), 2)));
        assert!(vm.skip_next_trace_record);
        assert!(matches!(
            vm.call("succeeding", Vec::new()),
            Ok(Value::Int(7))
        ));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn jit_guard_exit_resumes_at_bailout_ip() {
        use crate::jit::trace::{Trace, TraceOp};
        use crate::jit::TraceId;

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
            .compile_trace(&trace, TraceId(0), None, Vec::new())
            .unwrap();

        let mut vm = VM::new();
        vm.load_functions(vec![function]);
        vm.jit.store_root_trace(0, 0, compiled);

        assert!(matches!(
            vm.call("guard_exit", Vec::new()),
            Ok(Value::Int(99))
        ));
    }
}
