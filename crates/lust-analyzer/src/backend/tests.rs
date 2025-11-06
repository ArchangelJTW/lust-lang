use super::*;
use crate::analysis::{find_type_for_position, AnalysisSnapshot};
use crate::utils::{
    analyzer_lust_config, base_type_name, compute_line_offsets, offset_to_position,
    span_from_identifier,
};
use lust::modules::ModuleLoader;
use lust::TypeChecker;
use hashbrown::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tower_lsp::lsp_types::HoverContents;
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Self {
        let mut dir = std::env::temp_dir();
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        dir.push(format!("lust_lang_test_{unique}"));
        fs::create_dir_all(&dir).expect("create temp dir");
        Self { path: dir }
    }

    fn path(&self) -> &Path {
        &self.path
    }

}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }

}

fn new_typechecker() -> TypeChecker {
    let config = analyzer_lust_config();
    TypeChecker::with_config(&config)
}

#[test]
fn struct_field_completion_basic() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
pub struct Point
x: int
y: int
end
pub function length(point: Point): int
local sum = point.x + point.y
return sum
end
"#;
    fs::write(&entry_path, source.trim_start()).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let text = fs::read_to_string(&entry_path).expect("read source");
    let line_offsets = compute_line_offsets(&text);
    let base_offset = text.find("point.x").expect("find pattern");
    let position = offset_to_position(&text, base_offset, &line_offsets);
    let ty = find_type_for_position(module, position)
        .expect("type for position")
        .1;
    let base_name = base_type_name(&ty).expect("base type name");
    let expected_qualified = format!("{}.Point", module_path);
    assert!(base_name == expected_qualified || base_name == "Point");
    let completions =
        struct_field_completions(&snapshot, Some(module_path.as_str()), &base_name, "");
    let labels: Vec<String> = completions.into_iter().map(|item| item.label).collect();
    assert!(labels.contains(&"x".to_string()));
    assert!(labels.contains(&"y".to_string()));
}

#[test]
fn struct_field_completion_after_dot_uses_previous_snapshot() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
x: int
y: int
end
impl Point
function new(x: int, y: int): Point
    return Point { x = x, y = y }
end
function distance_squared(self): int
    return self.x * self.x + self.y * self.y
end
function translate(self, dx: int, dy: int): Point
    return Point { x = self.x + dx, y = self.y + dy }
end
end
function demo(): int
local p2 = Point.new(10, 20)
local squared = p2:distance_squared()
return squared
end
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source = source.replace("p2:distance_squared()", "p2.");
    let line_offsets = compute_line_offsets(&edited_source);
    let cursor_offset = edited_source
        .find("p2.")
        .map(|idx| idx + "p2.".len())
        .expect("cursor offset");
    let context =
        analyze_member_method_context(&edited_source, cursor_offset).expect("context");
    assert!(matches!(context.kind, CompletionKind::Member));
    let object_start = context.object_start.expect("object start");
    let object_name = context.object_name.as_ref().expect("object name");
    let object_position = offset_to_position(&edited_source, object_start, &line_offsets);
    let target_line = object_position.line as usize + 1;
    let identifier_span =
        span_from_identifier(&edited_source, object_start, object_name, &line_offsets);
    let ty = type_for_identifier(
        module,
        &edited_source,
        &line_offsets,
        object_name,
        identifier_span,
        target_line,
    )
    .expect("identifier type");
    let base_name = base_type_name(&ty).expect("base type name");
    let expected_qualified = format!("{}.Point", module_path);
    assert!(base_name == expected_qualified || base_name == "Point");
    let completions = struct_field_completions(
        &snapshot,
        Some(module_path.as_str()),
        &base_name,
        &context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"x"));
    assert!(labels.contains(&"y"));
}

#[test]
fn struct_field_completion_for_expression_stmt_object() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let initial_source = r#"
struct Point
x: int
y: int
end
function demo(): int
local p1: Point = Point { x = 3, y = 4 }
return 0
end
"#;
    let initial_source = initial_source.trim_start().to_string();
    fs::write(&entry_path, &initial_source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source = r#"
struct Point
x: int
y: int
end
function demo(): int
local p1: Point = Point { x = 3, y = 4 }
p1.
end
"#
    .trim_start()
    .to_string();
    let line_offsets = compute_line_offsets(&edited_source);
    let cursor_offset = edited_source
        .find("p1.")
        .map(|idx| idx + "p1.".len())
        .expect("cursor offset");
    let context =
        analyze_member_method_context(&edited_source, cursor_offset).expect("context");
    assert!(matches!(context.kind, CompletionKind::Member));
    let object_start = context.object_start.expect("object start");
    let object_name = context.object_name.as_ref().expect("object name");
    let object_position = offset_to_position(&edited_source, object_start, &line_offsets);
    let object_line = object_position.line as usize + 1;
    let identifier_span =
        span_from_identifier(&edited_source, object_start, object_name, &line_offsets);
    let ty = type_for_identifier(
        module,
        &edited_source,
        &line_offsets,
        object_name,
        identifier_span,
        object_line,
    )
    .expect("identifier type");
    let base_name = base_type_name(&ty).expect("base type name");
    assert_eq!(base_name, format!("{}.Point", module_path));
    let completions = struct_field_completions(
        &snapshot,
        Some(module_path.as_str()),
        &base_name,
        &context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"x"));
    assert!(labels.contains(&"y"));
}

#[test]
fn struct_literal_expression_completion_after_dot() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
x: int
y: int
end
function demo(): Point
return Point { x = 1, y = 2 }
end
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source = source.replace("Point { x = 1, y = 2 }", "Point { x = 1, y = 2 }.");
    let trigger_index = edited_source.find("}.").expect("struct literal trigger") + 1;
    let cursor_offset = trigger_index + 1;
    let context =
        analyze_member_method_context(&edited_source, cursor_offset).expect("context");
    assert!(matches!(context.kind, CompletionKind::Member));
    assert!(context.object_start.is_none());
    assert!(context.object_name.is_none());
    let base_name = infer_struct_literal_base_name(
        &edited_source,
        context.object_end,
        module,
        &snapshot,
        Some(module_path.as_str()),
    )
    .expect("struct literal base type");
    let expected_qualified = format!("{}.Point", module_path);
    assert!(base_name == expected_qualified || base_name == "Point");
    let completions = struct_field_completions(
        &snapshot,
        Some(module_path.as_str()),
        &base_name,
        &context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"x"));
    assert!(labels.contains(&"y"));
}

#[test]
fn method_completion_for_constructor_call_expression() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
x: int
y: int
end
impl Point
function new(x: int, y: int): Point
    return Point { x = x, y = y }
end
function distance_squared(self): int
    return self.x * self.x + self.y * self.y
end
function translate(self, dx: int, dy: int): Point
    return Point { x = self.x + dx, y = self.y + dy }
end
end
function demo(): Point
return Point.new(1, 2)
end
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source = source.replace("Point.new(1, 2)", "Point.new(1, 2):");
    let pattern = "Point.new(1, 2):";
    let trigger_offset = edited_source
        .find(pattern)
        .map(|idx| idx + pattern.len() - 1)
        .expect("constructor trigger");
    let cursor_offset = trigger_offset + 1;
    let context =
        analyze_member_method_context(&edited_source, cursor_offset).expect("context");
    assert!(matches!(context.kind, CompletionKind::Method));
    assert!(context.object_start.is_none());
    assert!(context.object_name.is_none());
    let base_name =
        infer_constructor_call_base_name(&edited_source, context.object_end, module, &snapshot)
            .expect("constructor base type");
    let expected_qualified = format!("{}.Point", module_path);
    assert!(base_name == expected_qualified || base_name == "Point");
    let completions = instance_method_completions(
        &snapshot,
        &base_name,
        Some(module_path.as_str()),
        &context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"distance_squared"));
    assert!(labels.contains(&"translate"));
}

#[test]
fn method_completion_after_chained_method_call() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
x: int
y: int
end
impl Point
function new(x: int, y: int): Point
    return Point { x = x, y = y }
end
function distance_squared(self): int
    return self.x * self.x + self.y * self.y
end
function translate(self, dx: int, dy: int): Point
    return Point { x = self.x + dx, y = self.y + dy }
end
end
function demo(): Point
return Point.new(1, 2):translate(3, 4)
end
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source = source.replace("translate(3, 4)", "translate(3, 4):");
    let line_offsets = compute_line_offsets(&edited_source);
    let pattern = "translate(3, 4):";
    let trigger_offset = edited_source
        .find(pattern)
        .map(|idx| idx + pattern.len() - 1)
        .expect("method trigger");
    let cursor_offset = trigger_offset + 1;
    let context =
        analyze_member_method_context(&edited_source, cursor_offset).expect("context");
    assert!(matches!(context.kind, CompletionKind::Method));
    assert!(context.object_start.is_none());
    assert!(context.object_name.is_none());
    let base_name = resolve_base_type_name_for_context(
        module,
        &snapshot,
        Some(module_path.as_str()),
        &edited_source,
        &line_offsets,
        &context,
    )
    .expect("base type for chained call");
    let expected = format!("{}.Point", module_path);
    assert!(base_name == expected || base_name == "Point");
    let completions = instance_method_completions(
        &snapshot,
        &base_name,
        Some(module_path.as_str()),
        &context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"distance_squared"));
    assert!(labels.contains(&"translate"));
}

#[test]
fn struct_field_completion_after_chained_method_call() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
x: int
y: int
end
impl Point
function new(x: int, y: int): Point
    return Point { x = x, y = y }
end
function translate(self, dx: int, dy: int): Point
    return Point { x = self.x + dx, y = self.y + dy }
end
end
function demo(): Point
return Point.new(1, 2):translate(3, 4)
end
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source = source.replace("translate(3, 4)", "translate(3, 4).");
    let line_offsets = compute_line_offsets(&edited_source);
    let pattern = "translate(3, 4).";
    let trigger_offset = edited_source
        .find(pattern)
        .map(|idx| idx + pattern.len() - 1)
        .expect("field trigger");
    let cursor_offset = trigger_offset + 1;
    let context =
        analyze_member_method_context(&edited_source, cursor_offset).expect("context");
    assert!(matches!(context.kind, CompletionKind::Member));
    assert!(context.object_start.is_none());
    assert!(context.object_name.is_none());
    let base_name = resolve_base_type_name_for_context(
        module,
        &snapshot,
        Some(module_path.as_str()),
        &edited_source,
        &line_offsets,
        &context,
    )
    .expect("base type for chained call");
    let expected = format!("{}.Point", module_path);
    assert!(base_name == expected || base_name == "Point");
    let completions = struct_field_completions(
        &snapshot,
        Some(module_path.as_str()),
        &base_name,
        &context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"x"));
    assert!(labels.contains(&"y"));
}

#[test]
fn self_completion_uses_identifier_type() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
x: int
y: int
end
impl Point
function new(x: int, y: int): Point
    return Point { x = x, y = y }
end
function distance_squared(self): int
    local dx = self.x
    local dy = self.y
    return dx * dx + dy * dy
end
function translate(self, dx: int, dy: int): Point
    local dist = self:distance_squared()
    local offset = self.x
    return Point { x = self.x + dx, y = self.y + dy }
end
end
function demo(): int
local p = Point.new(0, 0)
return p:distance_squared()
end
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let field_source = source.replace("self.x", "self.");
    let field_offsets = compute_line_offsets(&field_source);
    let field_cursor = field_source
        .find("self.")
        .map(|idx| idx + "self.".len())
        .expect("self dot cursor");
    let field_context =
        analyze_member_method_context(&field_source, field_cursor).expect("field context");
    assert!(matches!(field_context.kind, CompletionKind::Member));
    let field_start = field_context.object_start.expect("field start");
    let field_name = field_context.object_name.as_ref().expect("field name");
    let field_position = offset_to_position(&field_source, field_start, &field_offsets);
    let field_line = field_position.line as usize + 1;
    let field_span =
        span_from_identifier(&field_source, field_start, field_name, &field_offsets);
    let field_type = type_for_identifier(
        module,
        &field_source,
        &field_offsets,
        field_name,
        field_span,
        field_line,
    )
    .expect("field identifier type");
    let field_base = base_type_name(&field_type).expect("field base");
    assert_eq!(field_base, format!("{}.Point", module_path));
    let field_completions = struct_field_completions(
        &snapshot,
        Some(module_path.as_str()),
        &field_base,
        &field_context.prefix,
    );
    let field_labels: Vec<_> = field_completions
        .iter()
        .map(|item| item.label.as_str())
        .collect();
    assert!(field_labels.contains(&"x"));
    assert!(field_labels.contains(&"y"));
    let method_source = source.replace("self:distance_squared()", "self:");
    let method_offsets = compute_line_offsets(&method_source);
    let method_cursor = method_source
        .find("self:")
        .map(|idx| idx + "self:".len())
        .expect("self colon cursor");
    let method_context =
        analyze_member_method_context(&method_source, method_cursor).expect("method context");
    assert!(matches!(method_context.kind, CompletionKind::Method));
    let method_start = method_context.object_start.expect("method start");
    let method_name = method_context.object_name.as_ref().expect("method name");
    let method_position = offset_to_position(&method_source, method_start, &method_offsets);
    let method_line = method_position.line as usize + 1;
    let method_span =
        span_from_identifier(&method_source, method_start, method_name, &method_offsets);
    let method_type = type_for_identifier(
        module,
        &method_source,
        &method_offsets,
        method_name,
        method_span,
        method_line,
    )
    .expect("method identifier type");
    let method_base = base_type_name(&method_type).expect("method base");
    assert_eq!(method_base, format!("{}.Point", module_path));
    let method_completions = instance_method_completions(
        &snapshot,
        &method_base,
        Some(module_path.as_str()),
        &method_context.prefix,
    );
    let method_labels: Vec<_> = method_completions
        .iter()
        .map(|item| item.label.as_str())
        .collect();
    assert!(method_labels.contains(&"distance_squared"));
    assert!(method_labels.contains(&"translate"));
}

#[test]
fn identifier_completion_includes_locals_and_builtins() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
function demo(value: int, times: int): int
local sum = value + times
return sum
end
"#;
    let source = source.trim_start();
    fs::write(&entry_path, source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let text = fs::read_to_string(&entry_path).expect("read source");
    let line_offsets = compute_line_offsets(&text);
    let offset = text.find("sum").expect("find return target") + "sum".len();
    let position = offset_to_position(&text, offset, &line_offsets);
    let completions = identifier_completions(module, &snapshot, &entry_path, position, "");
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"value"));
    assert!(labels.contains(&"times"));
    assert!(labels.contains(&"sum"));
    assert!(labels.contains(&"print"));
    assert!(labels.contains(&"task"));
    assert!(labels.contains(&"io"));
    assert!(labels.contains(&"os"));
}

#[test]
fn identifier_completion_after_concat_operator_uses_identifier_context() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
    x: int
    y: int
end
impl Point
    function new(x: int, y: int): Point
        return Point { x = x, y = y }
    end
end
function demo(): Point
    local my_point = Point.new(15, 5)
    println(my_point.x .. ", " .. my_point.x)
    return my_point
end
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source =
        source.replace(r#", " .. my_point.x)"#, r#", " .. my)"#);
    let line_offsets = compute_line_offsets(&edited_source);
    let marker = r#", " .. my"#;
    let start = edited_source
        .find(marker)
        .expect("find concatenation completion target");
    let offset = start + r#", " .. "#.len() + "my".len();
    let position = offset_to_position(&edited_source, offset, &line_offsets);
    assert!(analyze_member_method_context(&edited_source, offset).is_none());
    let identifier_context =
        analyze_identifier_context(&edited_source, offset).expect("identifier context");
    assert_eq!(identifier_context.prefix, "my");
    let completions = identifier_completions(
        module,
        &snapshot,
        &entry_path,
        position,
        &identifier_context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"my_point"));
}

#[test]
fn identifier_completion_after_function_call_retains_locals() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
function demo(): int
    local my_point = 42
    println(my_point)
    return my_point
end
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source = source.replace("    return my_point", "    my_po");
    let line_offsets = compute_line_offsets(&edited_source);
    let needle = "my_po";
    let start = edited_source
        .rfind(needle)
        .expect("find local prefix")
        + needle.len();
    let position = offset_to_position(&edited_source, start, &line_offsets);
    assert!(analyze_member_method_context(&edited_source, start).is_none());
    let identifier_context =
        analyze_identifier_context(&edited_source, start).expect("identifier context");
    assert_eq!(identifier_context.prefix, "my_po");
    let completions = identifier_completions(
        module,
        &snapshot,
        &entry_path,
        position,
        &identifier_context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"my_point"));
}

#[test]
fn identifier_completion_after_call_statement_in_script() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
    x: int
    y: int
end
impl Point
    function new(x: int, y: int): Point
        return Point { x = x, y = y }
    end
    function distance_squared(self): int
        local dx: int = self.x
        local dy: int = self.y
        return (dx * dx + dy * dy)
    end
    function translate(self, dx: int, dy: int): Point
        return Point { x = self.x + dx, y = self.y + dy }
    end
end
local my_point = Point.new(15, 5)
println(my_point.x)
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let edited_source = format!("{source}\n\nmy_po -- editing locals after call\n");
    let line_offsets = compute_line_offsets(&edited_source);
    let offset = edited_source
        .find("my_po")
        .expect("find identifier prefix")
        + "my_po".len();
    let position = offset_to_position(&edited_source, offset, &line_offsets);
    assert!(analyze_member_method_context(&edited_source, offset).is_none());
    let identifier_context =
        analyze_identifier_context(&edited_source, offset).expect("identifier context");
    assert_eq!(identifier_context.prefix, "my_po");
    let completions = identifier_completions(
        module,
        &snapshot,
        &entry_path,
        position,
        &identifier_context.prefix,
    );
    let labels: Vec<_> = completions.iter().map(|item| item.label.as_str()).collect();
    assert!(labels.contains(&"my_point"));
}

#[test]
fn hover_shows_method_signature_for_instance_call() {
    let tmp = TempDir::new();
    let entry_path = tmp.path().join("main.lust");
    let source = r#"
struct Point
    x: int
    y: int
end
impl Point
    function new(x: int, y: int): Point
        return Point { x = x, y = y }
    end
    function translate(self, dx: int, dy: int): Point
        return Point { x = self.x + dx, y = self.y + dy }
    end
end
local point = Point.new(0, 0)
local moved = point:translate(1, 1)
"#;
    let source = source.trim_start().to_string();
    fs::write(&entry_path, &source).expect("write source");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let module_path = snapshot
        .module_path_for_file(&entry_path)
        .expect("module path")
        .to_string();
    let line_offsets = compute_line_offsets(&source);
    let call_offset = source
        .rfind(":translate")
        .expect("find method call")
        + 1;
    let position = offset_to_position(&source, call_offset, &line_offsets);
    let hover = hover_for_method_call(
        &snapshot,
        module,
        Some(module_path.as_str()),
        &source,
        position,
        "translate",
    )
    .expect("method hover");
    let expected_position = offset_to_position(&source, call_offset, &line_offsets);
    let range = hover.range.expect("hover range");
    assert_eq!(range.start, expected_position);
    match &hover.contents {
        HoverContents::Markup(content) => {
            assert!(
                content
                    .value
                    .contains("fn translate(self, dx: int, dy: int) -> Point"),
                "hover content: {}",
                content.value
            );
            assert!(
                content.value.contains("Defined on `"),
                "hover content: {}",
                content.value
            );
        }

        _ => panic!("unexpected hover contents"),
    }

}

#[test]
fn module_path_completion_suggests_modules_and_exports() {
    let tmp = TempDir::new();
    let lib_dir = tmp.path().join("lib");
    fs::create_dir_all(&lib_dir).expect("create lib dir");
    let math_path = lib_dir.join("math.lust");
    let math_source = r#"
pub struct Point
x: int
y: int
end
pub function add(a: int, b: int): int
return a + b
end
"#;
    fs::write(&math_path, math_source.trim_start()).expect("write math");
    let entry_path = tmp.path().join("main.lust");
    let entry_source = r#"
use lib.math
function main(): int
return 0
end
"#;
    fs::write(&entry_path, entry_source.trim_start()).expect("write main");
    let mut loader = ModuleLoader::new(tmp.path());
    let program = loader
        .load_program_from_entry(entry_path.to_str().expect("utf8 path"))
        .expect("program");
    let mut imports_map = HashMap::new();
    for module in &program.modules {
        imports_map.insert(module.path.clone(), module.imports.clone());
    }

    let mut typechecker = new_typechecker();
    typechecker.set_imports_by_module(imports_map);
    typechecker
        .check_program(&program.modules)
        .expect("typecheck");
    let struct_defs = typechecker.struct_definitions();
    let enum_defs = typechecker.enum_definitions();
    let type_info = typechecker.take_type_info();
    let snapshot =
        AnalysisSnapshot::new(
            &program,
            type_info,
            &HashMap::new(),
            struct_defs,
            enum_defs,
            HashSet::new(),
        );
    let module = snapshot
        .module_for_file(&entry_path)
        .expect("module snapshot");
    let module_items = module_path_completions(&snapshot, module, &["lib".to_string()], "");
    let module_labels: Vec<_> = module_items
        .iter()
        .map(|item| item.label.as_str())
        .collect();
    assert!(module_labels.contains(&"math"));
    let export_items = module_path_completions(
        &snapshot,
        module,
        &["lib".to_string(), "math".to_string()],
        "",
    );
    let export_labels: Vec<_> = export_items
        .iter()
        .map(|item| item.label.as_str())
        .collect();
    assert!(export_labels.contains(&"Point"));
    assert!(export_labels.contains(&"add"));
}

#[test]
fn analyze_context_handles_cursor_on_trigger() {
    let text = "self.";
    let offset_on_trigger = text.find('.').expect("dot found");
    let offset_after_trigger = text.len();
    let ctx_on =
        analyze_member_method_context(text, offset_on_trigger).expect("context on trigger");
    assert!(ctx_on.prefix.is_empty());
    assert_eq!(ctx_on.object_name.as_deref(), Some("self"));
    let ctx_after =
        analyze_member_method_context(text, offset_after_trigger).expect("context after");
    assert!(ctx_after.prefix.is_empty());
    assert_eq!(ctx_after.object_name.as_deref(), Some("self"));
}
