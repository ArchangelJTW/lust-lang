use super::conversions::{FromLustValue, FunctionArgs, IntoTypedValue};
use super::program::{ensure_return_type, normalize_global_name, EmbeddedProgram};
use crate::ast::{Type, TypeKind};
use crate::bytecode::{FieldStorage, StructLayout, Value, ValueKey};
use crate::number::{LustFloat, LustInt};
use crate::typechecker::FunctionSignature;
use crate::{LustError, Result};
use hashbrown::HashMap;
use std::cell::{Ref, RefCell, RefMut};
use std::ops::Deref;
use std::rc::Rc;

pub struct TypedValue {
    value: Value,
    matcher: Box<dyn Fn(&Value, &Type) -> bool>,
    description: &'static str,
}

impl TypedValue {
    pub(crate) fn new<F>(value: Value, matcher: F, description: &'static str) -> Self
    where
        F: Fn(&Value, &Type) -> bool + 'static,
    {
        Self {
            value,
            matcher: Box::new(matcher),
            description,
        }
    }

    pub(crate) fn matches(&self, ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Union(types) => types.iter().any(|alt| (self.matcher)(&self.value, alt)),
            _ => (self.matcher)(&self.value, ty),
        }
    }

    pub(crate) fn description(&self) -> &'static str {
        self.description
    }

    pub(crate) fn into_value(self) -> Value {
        self.value
    }

    pub(crate) fn as_value(&self) -> &Value {
        &self.value
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

    pub(crate) fn into_parts(self) -> (String, TypedValue) {
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
    pub(crate) fn new(type_name: String, value: Value) -> Self {
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

    pub(crate) fn into_value(self) -> Value {
        self.value
    }
}

#[derive(Clone)]
pub struct FunctionHandle {
    value: Value,
}

impl FunctionHandle {
    pub(crate) fn is_callable_value(value: &Value) -> bool {
        matches!(
            value,
            Value::Function(_) | Value::Closure { .. } | Value::NativeFunction(_)
        )
    }

    pub(crate) fn new_unchecked(value: Value) -> Self {
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
        program.vm().function_name(idx)
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
        program.vm_mut().call_value(self.as_value(), args)
    }

    pub fn call_typed<Args, R>(&self, program: &mut EmbeddedProgram, args: Args) -> Result<R>
    where
        Args: FunctionArgs,
        R: FromLustValue,
    {
        let program_ref: &EmbeddedProgram = &*program;
        let values = args.into_values();
        if let Some((name, signature)) = self.signature(program_ref) {
            Args::validate_signature(name, &signature.params)?;
            ensure_return_type::<R>(name, &signature.return_type)?;
            let value = program.vm_mut().call_value(self.as_value(), values)?;
            R::from_value(value)
        } else {
            let value = program.vm_mut().call_value(self.as_value(), values)?;
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
    pub(crate) fn new(value: ValueRef<'a>) -> Self {
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
    pub(crate) fn from_rc(inner: Rc<RefCell<Vec<Value>>>) -> Self {
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

    pub(crate) fn into_value(self) -> Value {
        Value::Array(self.inner)
    }
}

#[derive(Clone)]
pub struct MapHandle {
    inner: Rc<RefCell<HashMap<ValueKey, Value>>>,
}

impl MapHandle {
    pub(crate) fn from_rc(inner: Rc<RefCell<HashMap<ValueKey, Value>>>) -> Self {
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

    pub(crate) fn into_value(self) -> Value {
        Value::Map(self.inner)
    }
}

#[derive(Clone)]
pub struct EnumInstance {
    type_name: String,
    variant: String,
    value: Value,
}

impl EnumInstance {
    pub(crate) fn new(type_name: String, variant: String, value: Value) -> Self {
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

    pub(crate) fn into_value(self) -> Value {
        self.value
    }
}

pub(crate) fn matches_lust_struct(value: &Value, ty: &Type) -> bool {
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

pub(crate) fn matches_lust_enum(value: &Value, ty: &Type) -> bool {
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

pub(crate) fn lust_type_names_match(value: &str, expected: &str) -> bool {
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

pub(crate) fn simple_type_name(name: &str) -> &str {
    name.rsplit(|c| c == '.' || c == ':').next().unwrap_or(name)
}

pub(crate) fn matches_array_type<F>(ty: &Type, matcher: &F) -> bool
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

pub(crate) fn matches_array_handle_type(ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::Array(_) | TypeKind::Unknown => true,
        TypeKind::Union(types) => types.iter().any(|alt| matches_array_handle_type(alt)),
        _ => false,
    }
}

pub(crate) fn matches_map_handle_type(ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::Map(_, _) | TypeKind::Unknown => true,
        TypeKind::Union(types) => types.iter().any(|alt| matches_map_handle_type(alt)),
        _ => false,
    }
}

pub(crate) fn matches_function_handle_type(ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::Function { .. } | TypeKind::Unknown => true,
        TypeKind::Union(types) => types.iter().any(|alt| matches_function_handle_type(alt)),
        _ => false,
    }
}

pub(crate) fn signatures_match(a: &FunctionSignature, b: &FunctionSignature) -> bool {
    if a.is_method != b.is_method || a.params.len() != b.params.len() {
        return false;
    }

    if a.return_type != b.return_type {
        return false;
    }

    a.params
        .iter()
        .zip(&b.params)
        .all(|(left, right)| left == right)
}

pub(crate) fn signature_to_string(signature: &FunctionSignature) -> String {
    let params = signature
        .params
        .iter()
        .map(|param| param.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("function({}) -> {}", params, signature.return_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Span;
    use crate::embed::{AsyncDriver, EmbeddedProgram, LustStructView};
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

        let signature = program.signature("main.add").expect("signature").clone();
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
