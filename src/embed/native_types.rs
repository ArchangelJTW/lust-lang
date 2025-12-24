use crate::ast::{
    EnumDef, EnumVariant, FieldOwnership, FunctionDef, FunctionParam, ImplBlock, Span, StructDef,
    StructField, TraitBound, TraitDef, TraitMethod, Type, TypeKind, Visibility,
};
use crate::typechecker::{FunctionSignature, TypeChecker};
use crate::vm::VM;
use crate::Result;
use hashbrown::HashMap;
use std::collections::BTreeMap;

/// Aggregates Lust type declarations that originate from Rust.
#[derive(Clone, Default)]
pub struct ExternRegistry {
    structs: Vec<StructDef>,
    enums: Vec<EnumDef>,
    traits: Vec<TraitDef>,
    impls: Vec<ImplBlock>,
    functions: Vec<FunctionDef>,
    constants: Vec<(String, Type)>,
}

impl ExternRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_struct(&mut self, def: StructDef) -> &mut Self {
        self.structs.push(def);
        self
    }

    pub fn add_enum(&mut self, def: EnumDef) -> &mut Self {
        self.enums.push(def);
        self
    }

    pub fn add_trait(&mut self, def: TraitDef) -> &mut Self {
        self.traits.push(def);
        self
    }

    pub fn add_impl(&mut self, impl_block: ImplBlock) -> &mut Self {
        self.impls.push(impl_block);
        self
    }

    pub fn add_function(&mut self, func: FunctionDef) -> &mut Self {
        self.functions.push(func);
        self
    }

    pub fn add_const(&mut self, name: impl Into<String>, ty: Type) -> &mut Self {
        self.constants.push((name.into(), ty));
        self
    }

    pub fn extend(&mut self, other: &ExternRegistry) {
        self.structs.extend(other.structs.iter().cloned());
        self.enums.extend(other.enums.iter().cloned());
        self.traits.extend(other.traits.iter().cloned());
        self.impls.extend(other.impls.iter().cloned());
        self.functions.extend(other.functions.iter().cloned());
        self.constants.extend(other.constants.iter().cloned());
    }

    pub fn register_with_typechecker(&self, checker: &mut TypeChecker) -> Result<()> {
        for def in &self.structs {
            checker.register_external_struct(def.clone())?;
        }
        for def in &self.enums {
            checker.register_external_enum(def.clone())?;
        }
        for def in &self.traits {
            checker.register_external_trait(def.clone())?;
        }
        for func in &self.functions {
            checker.register_external_function(function_signature_for(func))?;
        }
        for impl_block in &self.impls {
            checker.register_external_impl(impl_block.clone())?;
        }
        for (name, ty) in &self.constants {
            checker.register_external_constant(name.clone(), ty.clone())?;
        }
        Ok(())
    }

    pub fn register_with_vm(&self, vm: &mut VM) {
        self.register_struct_layouts(vm);
        self.register_type_stubs(vm);
    }

    pub fn register_struct_layouts(&self, vm: &mut VM) {
        let prefix = vm.export_prefix();
        let mut struct_map: HashMap<String, StructDef> = HashMap::new();
        for def in &self.structs {
            let canonical = canonicalize_struct(def, prefix.as_deref());
            struct_map.insert(canonical.name.clone(), canonical);
        }
        if !struct_map.is_empty() {
            vm.register_structs(&struct_map);
        }
    }

    pub fn register_type_stubs(&self, vm: &mut VM) {
        if self.structs.is_empty()
            && self.enums.is_empty()
            && self.traits.is_empty()
            && self.impls.is_empty()
            && self.constants.is_empty()
        {
            return;
        }
        let prefix = vm.export_prefix();
        vm.register_type_stubs(self.module_stubs_with_prefix(prefix.as_deref()));
    }

    pub fn module_stubs(&self) -> Vec<ModuleStub> {
        self.module_stubs_with_prefix(None)
    }

    fn module_stubs_with_prefix(&self, prefix: Option<&str>) -> Vec<ModuleStub> {
        let mut modules: BTreeMap<String, ModuleStub> = BTreeMap::new();
        for def in &self.structs {
            let canonical = canonicalize_struct(def, prefix);
            if let Some((module, name)) = module_and_name(&canonical.name) {
                let entry = modules.entry(module.clone()).or_insert_with(|| ModuleStub {
                    module,
                    ..ModuleStub::default()
                });
                entry.struct_defs.push(format_struct_def(name, &canonical));
            }
        }
        for def in &self.enums {
            let canonical = canonicalize_enum(def, prefix);
            if let Some((module, name)) = module_and_name(&canonical.name) {
                let entry = modules.entry(module.clone()).or_insert_with(|| ModuleStub {
                    module,
                    ..ModuleStub::default()
                });
                entry.enum_defs.push(format_enum_def(name, &canonical));
            }
        }
        for def in &self.traits {
            let canonical = canonicalize_trait(def, prefix);
            if let Some((module, name)) = module_and_name(&canonical.name) {
                let entry = modules.entry(module.clone()).or_insert_with(|| ModuleStub {
                    module,
                    ..ModuleStub::default()
                });
                entry.trait_defs.push(format_trait_def(name, &canonical));
            }
        }
        for (name, ty) in &self.constants {
            let canonical_name = canonicalize_simple_name(name, prefix);
            if let Some((module, short)) = module_and_name(&canonical_name) {
                let entry = modules.entry(module.clone()).or_insert_with(|| ModuleStub {
                    module,
                    ..ModuleStub::default()
                });
                entry
                    .const_defs
                    .push(format_const_def(short, &canonicalize_type(ty, prefix)));
            }
        }
        for def in &self.impls {
            let canonical = canonicalize_impl(def, prefix);
            if let Some(module) = impl_module(&canonical) {
                modules.entry(module.clone()).or_insert_with(|| ModuleStub {
                    module,
                    ..ModuleStub::default()
                });
            }
        }
        modules.into_values().collect()
    }

    pub fn structs(&self) -> impl Iterator<Item = &StructDef> {
        self.structs.iter()
    }

    pub fn enums(&self) -> impl Iterator<Item = &EnumDef> {
        self.enums.iter()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModuleStub {
    pub module: String,
    pub struct_defs: Vec<String>,
    pub enum_defs: Vec<String>,
    pub trait_defs: Vec<String>,
    pub const_defs: Vec<String>,
}

impl ModuleStub {
    pub fn is_empty(&self) -> bool {
        self.struct_defs.is_empty()
            && self.enum_defs.is_empty()
            && self.trait_defs.is_empty()
            && self.const_defs.is_empty()
    }
}

#[derive(Clone)]
pub struct StructBuilder {
    def: StructDef,
}

impl StructBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        StructBuilder {
            def: StructDef {
                name: name.into(),
                type_params: Vec::new(),
                trait_bounds: Vec::new(),
                fields: Vec::new(),
                visibility: Visibility::Public,
            },
        }
    }

    pub fn visibility(mut self, visibility: Visibility) -> Self {
        self.def.visibility = visibility;
        self
    }

    pub fn type_param(mut self, name: impl Into<String>) -> Self {
        self.def.type_params.push(name.into());
        self
    }

    pub fn trait_bound(mut self, bound: TraitBound) -> Self {
        self.def.trait_bounds.push(bound);
        self
    }

    pub fn field(mut self, field: StructField) -> Self {
        self.def.fields.push(field);
        self
    }

    pub fn finish(self) -> StructDef {
        self.def
    }
}

#[derive(Clone)]
pub struct EnumBuilder {
    def: EnumDef,
}

impl EnumBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        EnumBuilder {
            def: EnumDef {
                name: name.into(),
                type_params: Vec::new(),
                trait_bounds: Vec::new(),
                variants: Vec::new(),
                visibility: Visibility::Public,
            },
        }
    }

    pub fn visibility(mut self, visibility: Visibility) -> Self {
        self.def.visibility = visibility;
        self
    }

    pub fn type_param(mut self, name: impl Into<String>) -> Self {
        self.def.type_params.push(name.into());
        self
    }

    pub fn trait_bound(mut self, bound: TraitBound) -> Self {
        self.def.trait_bounds.push(bound);
        self
    }

    pub fn variant(mut self, variant: EnumVariant) -> Self {
        self.def.variants.push(variant);
        self
    }

    pub fn finish(self) -> EnumDef {
        self.def
    }
}

#[derive(Clone)]
pub struct TraitBuilder {
    def: TraitDef,
}

impl TraitBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        TraitBuilder {
            def: TraitDef {
                name: name.into(),
                type_params: Vec::new(),
                methods: Vec::new(),
                visibility: Visibility::Public,
            },
        }
    }

    pub fn visibility(mut self, visibility: Visibility) -> Self {
        self.def.visibility = visibility;
        self
    }

    pub fn type_param(mut self, name: impl Into<String>) -> Self {
        self.def.type_params.push(name.into());
        self
    }

    pub fn method(mut self, method: TraitMethod) -> Self {
        self.def.methods.push(method);
        self
    }

    pub fn finish(self) -> TraitDef {
        self.def
    }
}

#[derive(Clone)]
pub struct ImplBuilder {
    block: ImplBlock,
}

impl ImplBuilder {
    pub fn new(target_type: Type) -> Self {
        ImplBuilder {
            block: ImplBlock {
                type_params: Vec::new(),
                trait_name: None,
                target_type,
                methods: Vec::new(),
                where_clause: Vec::new(),
            },
        }
    }

    pub fn for_trait(mut self, trait_name: impl Into<String>) -> Self {
        self.block.trait_name = Some(trait_name.into());
        self
    }

    pub fn type_param(mut self, name: impl Into<String>) -> Self {
        self.block.type_params.push(name.into());
        self
    }

    pub fn where_bound(mut self, bound: TraitBound) -> Self {
        self.block.where_clause.push(bound);
        self
    }

    pub fn method(mut self, method: FunctionDef) -> Self {
        self.block.methods.push(method);
        self
    }

    pub fn finish(self) -> ImplBlock {
        self.block
    }
}

#[derive(Clone)]
pub struct TraitMethodBuilder {
    method: TraitMethod,
}

impl TraitMethodBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        TraitMethodBuilder {
            method: TraitMethod {
                name: name.into(),
                type_params: Vec::new(),
                params: Vec::new(),
                return_type: None,
                default_impl: None,
            },
        }
    }

    pub fn type_param(mut self, name: impl Into<String>) -> Self {
        self.method.type_params.push(name.into());
        self
    }

    pub fn param(mut self, param: FunctionParam) -> Self {
        self.method.params.push(param);
        self
    }

    pub fn return_type(mut self, ty: Type) -> Self {
        self.method.return_type = Some(ty);
        self
    }

    pub fn finish(self) -> TraitMethod {
        self.method
    }
}

#[derive(Clone)]
pub struct FunctionBuilder {
    function: FunctionDef,
}

impl FunctionBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        FunctionBuilder {
            function: FunctionDef {
                name,
                type_params: Vec::new(),
                trait_bounds: Vec::new(),
                params: Vec::new(),
                return_type: None,
                body: Vec::new(),
                is_method: false,
                visibility: Visibility::Public,
            },
        }
    }

    pub fn visibility(mut self, visibility: Visibility) -> Self {
        self.function.visibility = visibility;
        self
    }

    pub fn type_param(mut self, name: impl Into<String>) -> Self {
        self.function.type_params.push(name.into());
        self
    }

    pub fn trait_bound(mut self, bound: TraitBound) -> Self {
        self.function.trait_bounds.push(bound);
        self
    }

    pub fn param(mut self, param: FunctionParam) -> Self {
        if param.is_self {
            self.function.is_method = true;
        }
        self.function.params.push(param);
        self
    }

    pub fn return_type(mut self, ty: Type) -> Self {
        self.function.return_type = Some(ty);
        self
    }

    pub fn finish(self) -> FunctionDef {
        self.function
    }
}

pub fn struct_field_decl(name: impl Into<String>, ty: Type) -> StructField {
    StructField {
        name: name.into(),
        ty,
        visibility: Visibility::Public,
        ownership: FieldOwnership::Strong,
        weak_target: None,
    }
}

pub fn weak_struct_field_decl(name: impl Into<String>, ty: Type) -> StructField {
    let span = ty.span;
    let option = Type::new(TypeKind::Option(Box::new(ty.clone())), span);
    StructField {
        name: name.into(),
        ty: option,
        visibility: Visibility::Public,
        ownership: FieldOwnership::Weak,
        weak_target: Some(ty),
    }
}

pub fn private_struct_field_decl(name: impl Into<String>, ty: Type) -> StructField {
    StructField {
        name: name.into(),
        ty,
        visibility: Visibility::Private,
        ownership: FieldOwnership::Strong,
        weak_target: None,
    }
}

pub fn enum_variant(name: impl Into<String>) -> EnumVariant {
    EnumVariant {
        name: name.into(),
        fields: None,
    }
}

pub fn enum_variant_with(
    name: impl Into<String>,
    fields: impl IntoIterator<Item = Type>,
) -> EnumVariant {
    EnumVariant {
        name: name.into(),
        fields: Some(fields.into_iter().collect()),
    }
}

pub fn trait_bound(
    param: impl Into<String>,
    traits: impl IntoIterator<Item = String>,
) -> TraitBound {
    TraitBound {
        type_param: param.into(),
        traits: traits.into_iter().collect(),
    }
}

pub fn function_param(name: impl Into<String>, ty: Type) -> FunctionParam {
    FunctionParam {
        name: name.into(),
        ty,
        is_self: false,
    }
}

pub fn self_param(ty: Option<Type>) -> FunctionParam {
    FunctionParam {
        name: "self".to_string(),
        ty: ty.unwrap_or_else(|| Type::new(TypeKind::Infer, Span::dummy())),
        is_self: true,
    }
}

pub fn type_named(name: impl Into<String>) -> Type {
    Type::new(TypeKind::Named(name.into()), Span::dummy())
}

pub fn type_unit() -> Type {
    Type::new(TypeKind::Unit, Span::dummy())
}

pub fn type_unknown() -> Type {
    Type::new(TypeKind::Unknown, Span::dummy())
}

fn function_signature_for(func: &FunctionDef) -> (String, FunctionSignature) {
    let params: Vec<Type> = func.params.iter().map(|p| p.ty.clone()).collect();
    let return_type = func
        .return_type
        .clone()
        .unwrap_or_else(|| Type::new(TypeKind::Unit, Span::dummy()));
    (
        func.name.clone(),
        FunctionSignature {
            params,
            return_type,
            is_method: func.is_method,
        },
    )
}

fn module_and_name(name: &str) -> Option<(String, String)> {
    if let Some((module, rest)) = name.rsplit_once("::") {
        return Some((module.replace("::", "."), rest.to_string()));
    }
    if let Some((module, rest)) = name.rsplit_once('.') {
        return Some((module.to_string(), rest.to_string()));
    }
    None
}

fn impl_module(impl_block: &ImplBlock) -> Option<String> {
    match &impl_block.target_type.kind {
        TypeKind::Named(name) => module_and_name(name).map(|(module, _)| module),
        TypeKind::GenericInstance { name, .. } => module_and_name(name).map(|(module, _)| module),
        _ => None,
    }
}

fn format_struct_def(name: String, def: &StructDef) -> String {
    let mut out = String::new();
    if matches!(def.visibility, Visibility::Public) {
        out.push_str("pub ");
    }
    out.push_str("struct ");
    out.push_str(&name);
    if !def.type_params.is_empty() {
        out.push('<');
        out.push_str(&def.type_params.join(", "));
        out.push('>');
    }
    out.push('\n');
    for field in &def.fields {
        out.push_str("    ");
        if matches!(field.visibility, Visibility::Public) {
            out.push_str("pub ");
        }
        out.push_str(&field.name);
        out.push_str(": ");
        match field.ownership {
            FieldOwnership::Weak => {
                out.push_str("ref ");
                if let Some(target) = &field.weak_target {
                    out.push_str(&format_type(target));
                } else {
                    out.push_str(&format_type(&field.ty));
                }
            }
            FieldOwnership::Strong => {
                out.push_str(&format_type(&field.ty));
            }
        }
        out.push('\n');
    }
    out.push_str("end\n");
    out
}

fn format_enum_def(name: String, def: &EnumDef) -> String {
    let mut out = String::new();
    if matches!(def.visibility, Visibility::Public) {
        out.push_str("pub ");
    }
    out.push_str("enum ");
    out.push_str(&name);
    if !def.type_params.is_empty() {
        out.push('<');
        out.push_str(&def.type_params.join(", "));
        out.push('>');
    }
    out.push('\n');
    for variant in &def.variants {
        out.push_str("    ");
        out.push_str(&variant.name);
        if let Some(fields) = &variant.fields {
            let parts: Vec<String> = fields.iter().map(format_type).collect();
            out.push('(');
            out.push_str(&parts.join(", "));
            out.push(')');
        }
        out.push('\n');
    }
    out.push_str("end\n");
    out
}

fn format_trait_def(name: String, def: &TraitDef) -> String {
    let mut out = String::new();
    if matches!(def.visibility, Visibility::Public) {
        out.push_str("pub ");
    }
    out.push_str("trait ");
    out.push_str(&name);
    if !def.type_params.is_empty() {
        out.push('<');
        out.push_str(&def.type_params.join(", "));
        out.push('>');
    }
    out.push('\n');
    for method in &def.methods {
        out.push_str("    ");
        out.push_str(&format_trait_method(method));
        out.push('\n');
    }
    out.push_str("end\n");
    out
}

fn format_const_def(name: String, ty: &Type) -> String {
    let mut out = String::new();
    out.push_str("pub const ");
    out.push_str(&name);
    out.push_str(": ");
    out.push_str(&format_type(ty));
    out.push('\n');
    out
}

fn format_trait_method(method: &TraitMethod) -> String {
    let mut out = String::new();
    out.push_str("function ");
    out.push_str(&method.name);
    if !method.type_params.is_empty() {
        out.push('<');
        out.push_str(&method.type_params.join(", "));
        out.push('>');
    }
    out.push('(');
    out.push_str(
        &method
            .params
            .iter()
            .map(format_param_signature)
            .collect::<Vec<_>>()
            .join(", "),
    );
    out.push(')');
    if let Some(ret) = &method.return_type {
        if !matches!(ret.kind, TypeKind::Unit) {
            out.push_str(": ");
            out.push_str(&format_type(ret));
        }
    }
    out
}

fn format_param_signature(param: &FunctionParam) -> String {
    if param.is_self {
        return "self".to_string();
    }
    if matches!(param.ty.kind, TypeKind::Infer) {
        param.name.clone()
    } else {
        format!("{}: {}", param.name, format_type(&param.ty))
    }
}

fn format_type(ty: &Type) -> String {
    match &ty.kind {
        TypeKind::Named(name) => {
            let (_, simple) =
                module_and_name(name).unwrap_or_else(|| (String::new(), name.clone()));
            if simple.is_empty() {
                name.clone()
            } else {
                simple
            }
        }
        TypeKind::GenericInstance { name, type_args } => {
            let (_, simple) =
                module_and_name(name).unwrap_or_else(|| (String::new(), name.clone()));
            let args = type_args
                .iter()
                .map(format_type)
                .collect::<Vec<_>>()
                .join(", ");
            format!("{}<{}>", simple, args)
        }
        _ => format!("{}", ty),
    }
}

fn canonicalize_struct(def: &StructDef, prefix: Option<&str>) -> StructDef {
    let mut cloned = def.clone();
    cloned.name = canonicalize_simple_name(&cloned.name, prefix);
    for field in &mut cloned.fields {
        field.ty = canonicalize_type(&field.ty, prefix);
        if let Some(target) = &field.weak_target {
            field.weak_target = Some(canonicalize_type(target, prefix));
        }
    }
    cloned
}

fn canonicalize_enum(def: &EnumDef, prefix: Option<&str>) -> EnumDef {
    let mut cloned = def.clone();
    cloned.name = canonicalize_simple_name(&cloned.name, prefix);
    for variant in &mut cloned.variants {
        if let Some(fields) = &mut variant.fields {
            for field in fields {
                *field = canonicalize_type(field, prefix);
            }
        }
    }
    cloned
}

fn canonicalize_trait(def: &TraitDef, prefix: Option<&str>) -> TraitDef {
    let mut cloned = def.clone();
    cloned.name = canonicalize_simple_name(&cloned.name, prefix);
    for method in &mut cloned.methods {
        for param in &mut method.params {
            param.ty = canonicalize_type(&param.ty, prefix);
        }
        if let Some(ret) = method.return_type.clone() {
            method.return_type = Some(canonicalize_type(&ret, prefix));
        }
    }
    cloned
}

fn canonicalize_impl(def: &ImplBlock, prefix: Option<&str>) -> ImplBlock {
    let mut cloned = def.clone();
    cloned.target_type = canonicalize_type(&cloned.target_type, prefix);
    let struct_name = match &cloned.target_type.kind {
        TypeKind::Named(name) => name.clone(),
        TypeKind::GenericInstance { name, .. } => name.clone(),
        _ => cloned.target_type.to_string(),
    };
    if let Some(trait_name) = &cloned.trait_name {
        cloned.trait_name = Some(canonicalize_simple_name(trait_name, prefix));
    }
    for method in &mut cloned.methods {
        for param in &mut method.params {
            param.ty = canonicalize_type(&param.ty, prefix);
        }
        if let Some(ret) = method.return_type.clone() {
            method.return_type = Some(canonicalize_type(&ret, prefix));
        }
        if let Some((_, rest)) = method.name.split_once(':') {
            method.name = format!("{}:{}", struct_name, rest);
        } else if let Some((_, rest)) = method.name.split_once('.') {
            method.name = format!("{}.{}", struct_name, rest);
        } else {
            method.name = format!("{}:{}", struct_name, method.name);
        }
        #[cfg(debug_assertions)]
        eprintln!("canonicalized method name -> {}", method.name);
    }
    cloned
}

fn canonicalize_type(ty: &Type, prefix: Option<&str>) -> Type {
    use TypeKind as TK;
    let mut canonical = ty.clone();
    canonical.kind = match &ty.kind {
        TK::Named(name) => TK::Named(canonicalize_simple_name(name, prefix)),
        TK::GenericInstance { name, type_args } => TK::GenericInstance {
            name: canonicalize_simple_name(name, prefix),
            type_args: type_args
                .iter()
                .map(|arg| canonicalize_type(arg, prefix))
                .collect(),
        },
        TK::Array(inner) => TK::Array(Box::new(canonicalize_type(inner, prefix))),
        TK::Option(inner) => TK::Option(Box::new(canonicalize_type(inner, prefix))),
        TK::Result(ok, err) => TK::Result(
            Box::new(canonicalize_type(ok, prefix)),
            Box::new(canonicalize_type(err, prefix)),
        ),
        TK::Ref(inner) => TK::Ref(Box::new(canonicalize_type(inner, prefix))),
        TK::MutRef(inner) => TK::MutRef(Box::new(canonicalize_type(inner, prefix))),
        TK::Map(key, value) => TK::Map(
            Box::new(canonicalize_type(key, prefix)),
            Box::new(canonicalize_type(value, prefix)),
        ),
        TK::Tuple(elements) => TK::Tuple(
            elements
                .iter()
                .map(|element| canonicalize_type(element, prefix))
                .collect(),
        ),
        TK::Union(types) => TK::Union(
            types
                .iter()
                .map(|variant| canonicalize_type(variant, prefix))
                .collect(),
        ),
        other => other.clone(),
    };
    canonical
}

fn canonicalize_simple_name(name: &str, prefix: Option<&str>) -> String {
    if name.is_empty() {
        return name.to_string();
    }
    if is_qualified_name(name) {
        return name.replace("::", ".");
    }
    if let Some(prefix) = prefix {
        if prefix.is_empty() {
            return name.to_string();
        }
        return format!("{}.{}", prefix, name);
    }
    name.to_string()
}

fn is_qualified_name(name: &str) -> bool {
    name.contains('.') || name.contains("::")
}
