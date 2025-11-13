use crate::ast::Type;
use crate::jit;
use crate::number::{
    float_from_int, float_is_nan, float_to_hash_bits, int_from_float, int_from_usize, LustFloat,
    LustInt,
};
use crate::vm::{pop_vm_ptr, push_vm_ptr, VM};
use alloc::{
    borrow::ToOwned,
    format,
    rc::{Rc, Weak},
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::cell::RefCell;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::{ptr, slice, str};
use hashbrown::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskHandle(pub u64);
impl TaskHandle {
    pub fn id(&self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub enum ValueKey {
    Int(LustInt),
    Float(LustFloat),
    String(Rc<String>),
    Bool(bool),
}

impl ValueKey {
    pub fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Int(i) => Some(ValueKey::Int(*i)),
            Value::Float(f) => Some(ValueKey::Float(*f)),
            Value::String(s) => Some(ValueKey::String(s.clone())),
            Value::Bool(b) => Some(ValueKey::Bool(*b)),
            _ => None,
        }
    }

    pub fn string<S>(value: S) -> Self
    where
        S: Into<String>,
    {
        ValueKey::String(Rc::new(value.into()))
    }

    pub fn to_value(&self) -> Value {
        match self {
            ValueKey::Int(i) => Value::Int(*i),
            ValueKey::Float(f) => Value::Float(*f),
            ValueKey::String(s) => Value::String(s.clone()),
            ValueKey::Bool(b) => Value::Bool(*b),
        }
    }
}

impl PartialEq for ValueKey {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ValueKey::Int(a), ValueKey::Int(b)) => a == b,
            (ValueKey::Float(a), ValueKey::Float(b)) => {
                if float_is_nan(*a) && float_is_nan(*b) {
                    true
                } else {
                    a == b
                }
            }

            (ValueKey::String(a), ValueKey::String(b)) => a == b,
            (ValueKey::Bool(a), ValueKey::Bool(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for ValueKey {}
impl Hash for ValueKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ValueKey::Int(i) => {
                0u8.hash(state);
                i.hash(state);
            }

            ValueKey::Float(f) => {
                1u8.hash(state);
                if float_is_nan(*f) {
                    u64::MAX.hash(state);
                } else {
                    float_to_hash_bits(*f).hash(state);
                }
            }

            ValueKey::String(s) => {
                2u8.hash(state);
                s.hash(state);
            }

            ValueKey::Bool(b) => {
                3u8.hash(state);
                b.hash(state);
            }
        }
    }
}

impl fmt::Display for ValueKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValueKey::Int(i) => write!(f, "{}", i),
            ValueKey::Float(fl) => write!(f, "{}", fl),
            ValueKey::String(s) => write!(f, "{}", s),
            ValueKey::Bool(b) => write!(f, "{}", b),
        }
    }
}

impl From<LustInt> for ValueKey {
    fn from(value: LustInt) -> Self {
        ValueKey::Int(value)
    }
}

impl From<LustFloat> for ValueKey {
    fn from(value: LustFloat) -> Self {
        ValueKey::Float(value)
    }
}

impl From<bool> for ValueKey {
    fn from(value: bool) -> Self {
        ValueKey::Bool(value)
    }
}

impl From<String> for ValueKey {
    fn from(value: String) -> Self {
        ValueKey::String(Rc::new(value))
    }
}

impl From<&str> for ValueKey {
    fn from(value: &str) -> Self {
        ValueKey::String(Rc::new(value.to_owned()))
    }
}

impl From<Rc<String>> for ValueKey {
    fn from(value: Rc<String>) -> Self {
        ValueKey::String(value)
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTag {
    Nil,
    Bool,
    Int,
    Float,
    String,
    Array,
    Tuple,
    Map,
    Struct,
    Enum,
    Function,
    NativeFunction,
    Closure,
    Iterator,
    Task,
}

impl ValueTag {
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldStorage {
    Strong,
    Weak,
}

#[derive(Debug)]
pub struct StructLayout {
    name: String,
    field_names: Vec<Rc<String>>,
    field_lookup_ptr: HashMap<usize, usize>,
    field_lookup_str: HashMap<String, usize>,
    field_storage: Vec<FieldStorage>,
    field_types: Vec<Type>,
    weak_targets: Vec<Option<Type>>,
}

impl StructLayout {
    pub fn new(
        name: String,
        field_names: Vec<Rc<String>>,
        field_storage: Vec<FieldStorage>,
        field_types: Vec<Type>,
        weak_targets: Vec<Option<Type>>,
    ) -> Self {
        debug_assert_eq!(
            field_names.len(),
            field_storage.len(),
            "StructLayout::new expects field names and storage metadata to align"
        );
        debug_assert_eq!(
            field_names.len(),
            field_types.len(),
            "StructLayout::new expects field names and type metadata to align"
        );
        debug_assert_eq!(
            field_names.len(),
            weak_targets.len(),
            "StructLayout::new expects field names and weak target metadata to align"
        );
        let mut field_lookup_ptr = HashMap::with_capacity(field_names.len());
        let mut field_lookup_str = HashMap::with_capacity(field_names.len());
        for (index, field_name_rc) in field_names.iter().enumerate() {
            let ptr = Rc::as_ptr(field_name_rc) as usize;
            field_lookup_ptr.insert(ptr, index);
            field_lookup_str.insert((**field_name_rc).clone(), index);
        }

        Self {
            name,
            field_names,
            field_lookup_ptr,
            field_lookup_str,
            field_storage,
            field_types,
            weak_targets,
        }
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    pub fn field_names(&self) -> &[Rc<String>] {
        &self.field_names
    }

    #[inline]
    pub fn index_of_rc(&self, key: &Rc<String>) -> Option<usize> {
        let ptr = Rc::as_ptr(key) as usize;
        self.field_lookup_ptr
            .get(&ptr)
            .copied()
            .or_else(|| self.field_lookup_str.get(key.as_str()).copied())
    }

    #[inline]
    pub fn index_of_str(&self, key: &str) -> Option<usize> {
        self.field_lookup_str.get(key).copied()
    }

    #[inline]
    pub fn field_storage(&self, index: usize) -> FieldStorage {
        self.field_storage[index]
    }

    #[inline]
    pub fn field_type(&self, index: usize) -> &Type {
        &self.field_types[index]
    }

    #[inline]
    pub fn weak_target(&self, index: usize) -> Option<&Type> {
        self.weak_targets[index].as_ref()
    }

    #[inline]
    pub fn is_weak(&self, index: usize) -> bool {
        matches!(self.field_storage(index), FieldStorage::Weak)
    }

    pub fn canonicalize_field_value(&self, index: usize, value: Value) -> Result<Value, String> {
        match self.field_storage(index) {
            FieldStorage::Strong => Ok(value),
            FieldStorage::Weak => self.canonicalize_weak_field(index, value),
        }
    }

    pub fn materialize_field_value(&self, index: usize, value: Value) -> Value {
        match self.field_storage(index) {
            FieldStorage::Strong => value,
            FieldStorage::Weak => self.materialize_weak_field(value),
        }
    }

    fn canonicalize_weak_field(&self, index: usize, value: Value) -> Result<Value, String> {
        let field_name = self.field_names[index].as_str();
        match value {
            Value::Enum {
                enum_name,
                variant,
                values,
            } if enum_name == "Option" => {
                if variant == "Some" {
                    if let Some(inner_values) = values {
                        if let Some(inner) = inner_values.get(0) {
                            let coerced = self.to_weak_struct(field_name, inner.clone())?;
                            Ok(Value::enum_variant("Option", "Some", vec![coerced]))
                        } else {
                            Ok(Value::enum_unit("Option", "None"))
                        }
                    } else {
                        Ok(Value::enum_unit("Option", "None"))
                    }
                } else if variant == "None" {
                    Ok(Value::enum_unit("Option", "None"))
                } else {
                    Err(format!(
                        "Struct '{}' field '{}' uses 'ref' and must store Option values; received variant '{}'",
                        self.name, field_name, variant
                    ))
                }
            }

            Value::Nil => Ok(Value::enum_unit("Option", "None")),
            other => {
                let coerced = self.to_weak_struct(field_name, other)?;
                Ok(Value::enum_variant("Option", "Some", vec![coerced]))
            }
        }
    }

    fn materialize_weak_field(&self, value: Value) -> Value {
        match value {
            Value::Enum {
                enum_name,
                variant,
                values,
            } if enum_name == "Option" => {
                if variant == "Some" {
                    if let Some(inner_values) = values {
                        if let Some(inner) = inner_values.get(0) {
                            match inner {
                                Value::WeakStruct(ref weak) => {
                                    if let Some(upgraded) = weak.upgrade() {
                                        Value::enum_variant("Option", "Some", vec![upgraded])
                                    } else {
                                        Value::enum_unit("Option", "None")
                                    }
                                }

                                _ => Value::enum_variant("Option", "Some", vec![inner.clone()]),
                            }
                        } else {
                            Value::enum_unit("Option", "None")
                        }
                    } else {
                        Value::enum_unit("Option", "None")
                    }
                } else {
                    Value::enum_unit("Option", "None")
                }
            }

            Value::Nil => Value::enum_unit("Option", "None"),
            other => Value::enum_variant("Option", "Some", vec![other]),
        }
    }

    fn to_weak_struct(&self, field_name: &str, value: Value) -> Result<Value, String> {
        match value {
            Value::Struct {
                name,
                layout,
                fields,
            } => Ok(Value::WeakStruct(WeakStructRef::new(name, layout, &fields))),
            Value::WeakStruct(_) => Ok(value),
            other => {
                let ty = other.type_of();
                Err(format!(
                    "Struct '{}' field '{}' expects a struct reference but received value of type '{:?}'",
                    self.name, field_name, ty
                ))
            }
        }
    }
}

#[repr(C, u8)]
#[derive(Clone)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(LustInt),
    Float(LustFloat),
    String(Rc<String>),
    Array(Rc<RefCell<Vec<Value>>>),
    Tuple(Rc<Vec<Value>>),
    Map(Rc<RefCell<HashMap<ValueKey, Value>>>),
    Struct {
        name: String,
        layout: Rc<StructLayout>,
        fields: Rc<RefCell<Vec<Value>>>,
    },
    WeakStruct(WeakStructRef),
    Enum {
        enum_name: String,
        variant: String,
        values: Option<Rc<Vec<Value>>>,
    },
    Function(usize),
    NativeFunction(NativeFn),
    Closure {
        function_idx: usize,
        upvalues: Rc<Vec<Upvalue>>,
    },
    Iterator(Rc<RefCell<IteratorState>>),
    Task(TaskHandle),
}

#[derive(Debug, Clone)]
pub struct WeakStructRef {
    name: String,
    layout: Rc<StructLayout>,
    fields: Weak<RefCell<Vec<Value>>>,
}

impl WeakStructRef {
    pub fn new(name: String, layout: Rc<StructLayout>, fields: &Rc<RefCell<Vec<Value>>>) -> Self {
        Self {
            name,
            layout,
            fields: Rc::downgrade(fields),
        }
    }

    pub fn upgrade(&self) -> Option<Value> {
        self.fields.upgrade().map(|fields| Value::Struct {
            name: self.name.clone(),
            layout: self.layout.clone(),
            fields,
        })
    }

    pub fn struct_name(&self) -> &str {
        &self.name
    }
}

#[derive(Clone)]
pub enum IteratorState {
    Array {
        items: Vec<Value>,
        index: usize,
    },
    MapPairs {
        items: Vec<(ValueKey, Value)>,
        index: usize,
    },
}

#[derive(Clone)]
pub struct Upvalue {
    value: Rc<RefCell<Value>>,
}

impl Upvalue {
    pub fn new(value: Value) -> Self {
        Self {
            value: Rc::new(RefCell::new(value)),
        }
    }

    pub fn get(&self) -> Value {
        self.value.borrow().clone()
    }

    pub fn set(&self, value: Value) {
        *self.value.borrow_mut() = value;
    }
}

impl fmt::Debug for Upvalue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Upvalue({:?})", self.value.borrow())
    }
}

#[derive(Debug, Clone)]
pub enum NativeCallResult {
    Return(Value),
    Yield(Value),
    Stop(Value),
}

impl From<Value> for NativeCallResult {
    fn from(value: Value) -> Self {
        NativeCallResult::Return(value)
    }
}

pub type NativeFn = Rc<dyn Fn(&[Value]) -> Result<NativeCallResult, String>>;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Nil,
    Bool,
    Int,
    Float,
    String,
    Array,
    Tuple,
    Map,
    Struct,
    Enum,
    Function,
    NativeFunction,
    Closure,
    Iterator,
    Task,
}

impl Value {
    #[inline]
    pub fn tag(&self) -> ValueTag {
        match self {
            Value::Nil => ValueTag::Nil,
            Value::Bool(_) => ValueTag::Bool,
            Value::Int(_) => ValueTag::Int,
            Value::Float(_) => ValueTag::Float,
            Value::String(_) => ValueTag::String,
            Value::Array(_) => ValueTag::Array,
            Value::Tuple(_) => ValueTag::Tuple,
            Value::Map(_) => ValueTag::Map,
            Value::Struct { .. } | Value::WeakStruct(_) => ValueTag::Struct,
            Value::Enum { .. } => ValueTag::Enum,
            Value::Function(_) => ValueTag::Function,
            Value::NativeFunction(_) => ValueTag::NativeFunction,
            Value::Closure { .. } => ValueTag::Closure,
            Value::Iterator(_) => ValueTag::Iterator,
            Value::Task(_) => ValueTag::Task,
        }
    }

    pub fn type_of(&self) -> ValueType {
        match self {
            Value::Nil => ValueType::Nil,
            Value::Bool(_) => ValueType::Bool,
            Value::Int(_) => ValueType::Int,
            Value::Float(_) => ValueType::Float,
            Value::String(_) => ValueType::String,
            Value::Array(_) => ValueType::Array,
            Value::Tuple(_) => ValueType::Tuple,
            Value::Map(_) => ValueType::Map,
            Value::Struct { .. } | Value::WeakStruct(_) => ValueType::Struct,
            Value::Enum { .. } => ValueType::Enum,
            Value::Function(_) => ValueType::Function,
            Value::NativeFunction(_) => ValueType::NativeFunction,
            Value::Closure { .. } => ValueType::Closure,
            Value::Iterator(_) => ValueType::Iterator,
            Value::Task(_) => ValueType::Task,
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(false) => false,
            Value::Enum {
                enum_name, variant, ..
            } if enum_name == "Option" && variant == "None" => false,
            _ => true,
        }
    }

    pub fn to_bool(&self) -> bool {
        self.is_truthy()
    }

    pub fn as_int(&self) -> Option<LustInt> {
        match self {
            Value::Int(i) => Some(*i),
            Value::Float(f) => Some(int_from_float(*f)),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<LustFloat> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(float_from_int(*i)),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_string_rc(&self) -> Option<Rc<String>> {
        match self {
            Value::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_task_handle(&self) -> Option<TaskHandle> {
        match self {
            Value::Task(handle) => Some(*handle),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<Vec<Value>> {
        match self {
            Value::Array(arr) => Some(arr.borrow().clone()),
            _ => None,
        }
    }

    pub fn array_len(&self) -> Option<usize> {
        match self {
            Value::Array(arr) => Some(arr.borrow().len()),
            _ => None,
        }
    }

    pub fn array_get(&self, index: usize) -> Option<Value> {
        match self {
            Value::Array(arr) => arr.borrow().get(index).cloned(),
            _ => None,
        }
    }

    pub fn array_push(&self, value: Value) -> Result<(), String> {
        match self {
            Value::Array(arr) => {
                arr.borrow_mut().push(value);
                Ok(())
            }

            _ => Err("Cannot push to non-array".to_string()),
        }
    }

    pub fn array_pop(&self) -> Result<Option<Value>, String> {
        match self {
            Value::Array(arr) => Ok(arr.borrow_mut().pop()),
            _ => Err("Cannot pop from non-array".to_string()),
        }
    }

    pub fn as_map(&self) -> Option<HashMap<ValueKey, Value>> {
        match self {
            Value::Map(map) => Some(map.borrow().clone()),
            _ => None,
        }
    }

    pub fn map_get(&self, key: &ValueKey) -> Option<Value> {
        match self {
            Value::Map(map) => map.borrow().get(key).cloned(),
            _ => None,
        }
    }

    pub fn map_set(&self, key: ValueKey, value: Value) -> Result<(), String> {
        match self {
            Value::Map(map) => {
                map.borrow_mut().insert(key, value);
                Ok(())
            }

            _ => Err("Cannot set key on non-map".to_string()),
        }
    }

    pub fn map_has(&self, key: &ValueKey) -> Option<bool> {
        match self {
            Value::Map(map) => Some(map.borrow().contains_key(key)),
            _ => None,
        }
    }

    pub fn map_delete(&self, key: &ValueKey) -> Result<Option<Value>, String> {
        match self {
            Value::Map(map) => Ok(map.borrow_mut().remove(key)),
            _ => Err("Cannot delete key from non-map".to_string()),
        }
    }

    pub fn map_len(&self) -> Option<usize> {
        match self {
            Value::Map(map) => Some(map.borrow().len()),
            _ => None,
        }
    }

    pub fn string(s: impl Into<String>) -> Self {
        Value::String(Rc::new(s.into()))
    }

    pub fn array(values: Vec<Value>) -> Self {
        Value::Array(Rc::new(RefCell::new(values)))
    }

    pub fn tuple(values: Vec<Value>) -> Self {
        Value::Tuple(Rc::new(values))
    }

    pub fn tuple_len(&self) -> Option<usize> {
        match self {
            Value::Tuple(values) => Some(values.len()),
            _ => None,
        }
    }

    pub fn tuple_get(&self, index: usize) -> Option<Value> {
        match self {
            Value::Tuple(values) => values.get(index).cloned(),
            _ => None,
        }
    }

    pub fn map(entries: HashMap<ValueKey, Value>) -> Self {
        Value::Map(Rc::new(RefCell::new(entries)))
    }

    pub fn task(handle: TaskHandle) -> Self {
        Value::Task(handle)
    }

    pub fn struct_get_field_rc(&self, field: &Rc<String>) -> Option<Value> {
        match self {
            Value::Struct { layout, fields, .. } => layout
                .index_of_rc(field)
                .or_else(|| layout.index_of_str(field.as_str()))
                .or_else(|| {
                    layout
                        .field_names()
                        .iter()
                        .position(|name| name.as_str() == field.as_str())
                })
                .and_then(|idx| {
                    fields
                        .borrow()
                        .get(idx)
                        .cloned()
                        .map(|value| layout.materialize_field_value(idx, value))
                }),
            _ => None,
        }
    }

    pub fn struct_get_field(&self, field: &str) -> Option<Value> {
        match self {
            Value::Struct { layout, fields, .. } => layout.index_of_str(field).and_then(|idx| {
                fields
                    .borrow()
                    .get(idx)
                    .cloned()
                    .map(|value| layout.materialize_field_value(idx, value))
            }),
            _ => None,
        }
    }

    pub fn struct_get_field_indexed(&self, index: usize) -> Option<Value> {
        match self {
            Value::Struct { layout, fields, .. } => fields
                .borrow()
                .get(index)
                .cloned()
                .map(|value| layout.materialize_field_value(index, value)),
            _ => None,
        }
    }

    pub fn struct_set_field_rc(&self, field: &Rc<String>, value: Value) -> Result<(), String> {
        match self {
            Value::Struct { layout, .. } => {
                if let Some(index) = layout
                    .index_of_rc(field)
                    .or_else(|| layout.index_of_str(field.as_str()))
                {
                    self.struct_set_field_indexed(index, value)
                } else {
                    Err(format!(
                        "Struct '{}' has no field '{}'",
                        layout.name(),
                        field.as_str()
                    ))
                }
            }

            _ => Err("Attempted to set field on non-struct value".to_string()),
        }
    }

    pub fn struct_set_field(&self, field: &str, value: Value) -> Result<(), String> {
        match self {
            Value::Struct { layout, .. } => {
                if let Some(index) = layout.index_of_str(field) {
                    self.struct_set_field_indexed(index, value)
                } else {
                    Err(format!(
                        "Struct '{}' has no field '{}'",
                        layout.name(),
                        field
                    ))
                }
            }

            _ => Err("Attempted to set field on non-struct value".to_string()),
        }
    }

    pub fn struct_set_field_indexed(&self, index: usize, value: Value) -> Result<(), String> {
        match self {
            Value::Struct {
                name,
                layout,
                fields,
            } => {
                let mut borrowed = fields.borrow_mut();
                if index < borrowed.len() {
                    let canonical = layout.canonicalize_field_value(index, value)?;
                    borrowed[index] = canonical;
                    Ok(())
                } else {
                    Err(format!(
                        "Struct '{}' field index {} out of bounds (len {})",
                        name,
                        index,
                        borrowed.len()
                    ))
                }
            }

            _ => Err("Attempted to set field on non-struct value".to_string()),
        }
    }

    pub fn enum_unit(enum_name: impl Into<String>, variant: impl Into<String>) -> Self {
        Value::Enum {
            enum_name: enum_name.into(),
            variant: variant.into(),
            values: None,
        }
    }

    pub fn enum_variant(
        enum_name: impl Into<String>,
        variant: impl Into<String>,
        values: Vec<Value>,
    ) -> Self {
        Value::Enum {
            enum_name: enum_name.into(),
            variant: variant.into(),
            values: Some(Rc::new(values)),
        }
    }

    pub fn as_enum(&self) -> Option<(&str, &str, Option<&[Value]>)> {
        match self {
            Value::Enum {
                enum_name,
                variant,
                values,
            } => Some((
                enum_name.as_str(),
                variant.as_str(),
                values.as_ref().map(|v| v.as_slice()),
            )),
            _ => None,
        }
    }

    pub fn is_enum_variant(&self, enum_name: &str, variant: &str) -> bool {
        match self {
            Value::Enum {
                enum_name: en,
                variant: v,
                ..
            } => (enum_name.is_empty() || en == enum_name) && v == variant,
            _ => false,
        }
    }

    pub fn some(value: Value) -> Self {
        Value::enum_variant("Option", "Some", vec![value])
    }

    pub fn none() -> Self {
        Value::enum_unit("Option", "None")
    }

    pub fn ok(value: Value) -> Self {
        Value::enum_variant("Result", "Ok", vec![value])
    }

    pub fn err(error: Value) -> Self {
        Value::enum_variant("Result", "Err", vec![error])
    }

    pub fn to_string(&self) -> String {
        format!("{}", self)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "Nil"),
            Value::Bool(b) => write!(f, "Bool({})", b),
            Value::Int(i) => write!(f, "Int({})", i),
            Value::Float(fl) => write!(f, "Float({})", fl),
            Value::String(s) => write!(f, "String({:?})", s),
            Value::Array(arr) => write!(f, "Array({:?})", arr.borrow()),
            Value::Tuple(values) => write!(f, "Tuple({:?})", values),
            Value::Map(map) => write!(f, "Map({:?})", map.borrow()),
            Value::Struct {
                name,
                layout,
                fields,
            } => {
                let borrowed = fields.borrow();
                let mut display_fields = Vec::with_capacity(borrowed.len());
                for (idx, field_name) in layout.field_names().iter().enumerate() {
                    let value = borrowed.get(idx).cloned().unwrap_or(Value::Nil);
                    display_fields.push((field_name.as_str().to_string(), value));
                }

                write!(
                    f,
                    "Struct {{ name: {:?}, fields: {:?} }}",
                    name, display_fields
                )
            }

            Value::WeakStruct(weak) => {
                if let Some(upgraded) = weak.upgrade() {
                    write!(f, "WeakStruct({:?})", upgraded)
                } else {
                    write!(f, "WeakStruct(<dangling>)")
                }
            }

            Value::Enum {
                enum_name,
                variant,
                values,
            } => {
                write!(
                    f,
                    "Enum {{ enum: {:?}, variant: {:?}, values: {:?} }}",
                    enum_name, variant, values
                )
            }

            Value::Function(idx) => write!(f, "Function({})", idx),
            Value::NativeFunction(_) => write!(f, "NativeFunction(<fn>)"),
            Value::Closure {
                function_idx,
                upvalues,
            } => {
                write!(
                    f,
                    "Closure {{ function: {}, upvalues: {:?} }}",
                    function_idx, upvalues
                )
            }

            Value::Iterator(_) => write!(f, "Iterator(<state>)"),
            Value::Task(handle) => write!(f, "Task({})", handle.0),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Nil => write!(f, "nil"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Int(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "{}", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                let borrowed = arr.borrow();
                for (i, val) in borrowed.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    write!(f, "{}", val)?;
                }

                write!(f, "]")
            }

            Value::Tuple(values) => {
                write!(f, "(")?;
                for (i, val) in values.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    write!(f, "{}", val)?;
                }

                write!(f, ")")
            }

            Value::Map(map) => {
                write!(f, "{{")?;
                let borrowed = map.borrow();
                for (i, (k, v)) in borrowed.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    write!(f, "{}: {}", k, v)?;
                }

                write!(f, "}}")
            }

            Value::Struct {
                name,
                layout,
                fields,
            } => {
                let borrowed = fields.borrow();
                write!(f, "{} {{", name)?;
                for (i, field_name) in layout.field_names().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    let value = borrowed.get(i).unwrap_or(&Value::Nil);
                    write!(f, "{}: {}", field_name, value)?;
                }

                write!(f, "}}")
            }

            Value::WeakStruct(weak) => {
                if let Some(strong) = weak.upgrade() {
                    strong.fmt(f)
                } else {
                    write!(f, "nil")
                }
            }

            Value::Enum {
                enum_name,
                variant,
                values,
            } => {
                write!(f, "{}.{}", enum_name, variant)?;
                if let Some(vals) = values {
                    write!(f, "(")?;
                    for (i, val) in vals.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }

                        write!(f, "{}", val)?;
                    }

                    write!(f, ")")?;
                }

                Ok(())
            }

            Value::Function(idx) => write!(f, "<function@{}>", idx),
            Value::NativeFunction(_) => write!(f, "<native function>"),
            Value::Closure { function_idx, .. } => write!(f, "<closure@{}>", function_idx),
            Value::Iterator(_) => write!(f, "<iterator>"),
            Value::Task(handle) => write!(f, "<task {}>", handle.0),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Nil, Value::Nil) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Array(a), Value::Array(b)) => *a.borrow() == *b.borrow(),
            (Value::Tuple(a), Value::Tuple(b)) => *a == *b,
            (Value::Map(a), Value::Map(b)) => *a.borrow() == *b.borrow(),
            (
                Value::Struct {
                    name: n1,
                    layout: l1,
                    fields: f1,
                },
                Value::Struct {
                    name: n2,
                    layout: l2,
                    fields: f2,
                },
            ) => {
                if n1 != n2 {
                    return false;
                }

                let borrowed_f1 = f1.borrow();
                let borrowed_f2 = f2.borrow();
                if borrowed_f1.len() != borrowed_f2.len() {
                    return false;
                }

                if Rc::ptr_eq(l1, l2) {
                    return borrowed_f1
                        .iter()
                        .zip(borrowed_f2.iter())
                        .all(|(a, b)| a == b);
                }

                l1.field_names()
                    .iter()
                    .enumerate()
                    .all(|(idx, field_name)| {
                        if let Some(other_idx) = l2.index_of_rc(field_name) {
                            borrowed_f1
                                .get(idx)
                                .zip(borrowed_f2.get(other_idx))
                                .map(|(a, b)| a == b)
                                .unwrap_or(false)
                        } else {
                            false
                        }
                    })
            }

            (Value::WeakStruct(a), Value::WeakStruct(b)) => match (a.upgrade(), b.upgrade()) {
                (Some(left), Some(right)) => left == right,
                (None, None) => true,
                _ => false,
            },
            (Value::WeakStruct(a), other) => a
                .upgrade()
                .map(|upgraded| upgraded == *other)
                .unwrap_or(matches!(other, Value::Nil)),
            (value, Value::WeakStruct(b)) => b
                .upgrade()
                .map(|upgraded| *value == upgraded)
                .unwrap_or(matches!(value, Value::Nil)),
            (
                Value::Enum {
                    enum_name: e1,
                    variant: v1,
                    values: vals1,
                },
                Value::Enum {
                    enum_name: e2,
                    variant: v2,
                    values: vals2,
                },
            ) => e1 == e2 && v1 == v2 && vals1 == vals2,
            (Value::Function(a), Value::Function(b)) => a == b,
            (
                Value::Closure {
                    function_idx: f1,
                    upvalues: u1,
                },
                Value::Closure {
                    function_idx: f2,
                    upvalues: u2,
                },
            ) => f1 == f2 && Rc::ptr_eq(u1, u2),
            (Value::Iterator(_), Value::Iterator(_)) => false,
            (Value::Task(a), Value::Task(b)) => a == b,
            _ => false,
        }
    }
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_array_get_safe(
    array_value_ptr: *const Value,
    index: i64,
    out: *mut Value,
) -> u8 {
    if array_value_ptr.is_null() || out.is_null() {
        eprintln!("❌ jit_array_get_safe: null pointer detected!");
        return 0;
    }

    let array_value = &*array_value_ptr;
    let arr = match array_value {
        Value::Array(arr) => arr,
        _ => {
            return 0;
        }
    };
    if index < 0 {
        return 0;
    }

    let idx = index as usize;
    let borrowed = match arr.try_borrow() {
        Ok(b) => b,
        Err(_) => {
            return 0;
        }
    };
    if idx >= borrowed.len() {
        return 0;
    }

    ptr::write(out, borrowed[idx].clone());
    1
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_array_len_safe(array_value_ptr: *const Value) -> i64 {
    if array_value_ptr.is_null() {
        return -1;
    }

    let array_value = &*array_value_ptr;
    match array_value {
        Value::Array(arr) => match arr.try_borrow() {
            Ok(borrowed) => int_from_usize(borrowed.len()),
            Err(_) => -1,
        },
        _ => -1,
    }
}

#[cfg(feature = "std")]
static JIT_NEW_ARRAY_COUNTER: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_new_array_safe(
    elements_ptr: *const Value,
    element_count: usize,
    out_ptr: *mut Value,
) -> u8 {
    let call_num = JIT_NEW_ARRAY_COUNTER.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    // jit::log(|| format!("jit_new_array_safe call #{}: ENTER - elements_ptr={:?}, count={}, out_ptr={:?}", call_num, elements_ptr, element_count, out_ptr));

    if out_ptr.is_null() {
        // jit::log(|| "jit_new_array_safe: out_ptr is null".to_string());
        return 0;
    }

    // jit::log(|| format!("jit_new_array_safe #{}: about to create Vec", call_num));
    let elements = if element_count == 0 {
        Vec::new()
    } else {
        if elements_ptr.is_null() {
            // jit::log(|| "jit_new_array_safe: elements_ptr is null but count > 0".to_string());
            return 0;
        }

        let slice = slice::from_raw_parts(elements_ptr, element_count);
        slice.to_vec()
    };

    // jit::log(|| format!("jit_new_array_safe #{}: about to call Value::array with {} elements", call_num, elements.len()));
    let array_value = Value::array(elements);
    // jit::log(|| format!("jit_new_array_safe #{}: about to write to out_ptr", call_num));
    ptr::write(out_ptr, array_value);
    // jit::log(|| format!("jit_new_array_safe #{}: EXIT - success, returning 1", call_num));
    1
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_array_push_safe(
    array_ptr: *const Value,
    value_ptr: *const Value,
) -> u8 {
    if array_ptr.is_null() || value_ptr.is_null() {
        return 0;
    }

    let array_value = &*array_ptr;
    let value = &*value_ptr;

    match array_value {
        Value::Array(arr) => {
            // Use unchecked borrow for maximum performance
            let cell_ptr = arr.as_ptr();
            (*cell_ptr).push(value.clone());
            1
        }
        _ => 0,
    }
}

/// Unbox Array<int> from Value to Vec<LustInt> for specialized JIT operations
/// IMPORTANT: This takes ownership of the array's data. The original Vec<Value> is emptied.
/// Returns 1 on success, 0 on failure
/// Outputs: vec_ptr (pointer to data), vec_len, vec_cap
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_unbox_array_int(
    array_value_ptr: *const Value,
    out_vec_ptr: *mut *mut LustInt,
    out_len: *mut usize,
    out_cap: *mut usize,
) -> u8 {
    if array_value_ptr.is_null() || out_vec_ptr.is_null() || out_len.is_null() || out_cap.is_null()
    {
        return 0;
    }

    let array_value = &*array_value_ptr;
    match array_value {
        Value::Array(arr_rc) => {
            // Get exclusive access to the inner vector
            let cell_ptr = arr_rc.as_ptr();
            let vec_ref = &mut *cell_ptr;

            // Take ownership of the Vec<Value> by swapping with empty vec
            let original_vec = core::mem::replace(vec_ref, Vec::new());

            // Convert Vec<Value> to Vec<LustInt>
            let mut specialized_vec: Vec<LustInt> = Vec::with_capacity(original_vec.len());
            for elem in original_vec.into_iter() {
                match elem {
                    Value::Int(i) => specialized_vec.push(i),
                    _ => {
                        // Type mismatch - restore original data and fail
                        // This shouldn't happen if types are correct, but be safe
                        return 0;
                    }
                }
            }

            // Extract Vec metadata
            let len = specialized_vec.len();
            let cap = specialized_vec.capacity();
            let ptr = specialized_vec.as_mut_ptr();

            // Prevent Vec from being dropped
            core::mem::forget(specialized_vec);

            // Write outputs
            ptr::write(out_vec_ptr, ptr);
            ptr::write(out_len, len);
            ptr::write(out_cap, cap);

            // Note: The original Rc<RefCell<Vec<Value>>> now contains an empty Vec
            // This is fine - when we rebox, we'll refill it

            1
        }
        _ => 0,
    }
}

/// Rebox Vec<LustInt> back to Array Value
/// IMPORTANT: Writes the specialized vec data back into the EXISTING Rc<RefCell<Vec<Value>>>
/// This ensures the original array is updated, not replaced
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_rebox_array_int(
    vec_ptr: *mut LustInt,
    vec_len: usize,
    vec_cap: usize,
    array_value_ptr: *mut Value,
) -> u8 {
    if vec_ptr.is_null() || array_value_ptr.is_null() {
        return 0;
    }

    // Reconstruct Vec<LustInt> from raw parts
    let specialized_vec = Vec::from_raw_parts(vec_ptr, vec_len, vec_cap);

    // Get the existing Array value
    let array_value = &mut *array_value_ptr;
    match array_value {
        Value::Array(arr_rc) => {
            // Get exclusive access to the inner vector (should be empty from unbox)
            let cell_ptr = arr_rc.as_ptr();
            let vec_ref = &mut *cell_ptr;

            // Convert Vec<LustInt> back to Vec<Value> and write into the existing RefCell
            *vec_ref = specialized_vec.into_iter().map(Value::Int).collect();

            1
        }
        _ => {
            // This shouldn't happen - the register should still contain the Array
            // But if it doesn't, create a new array
            let value_vec: Vec<Value> = specialized_vec.into_iter().map(Value::Int).collect();
            let array_value_new = Value::array(value_vec);
            ptr::write(array_value_ptr, array_value_new);
            1
        }
    }
}

/// Specialized push operation for Vec<LustInt>
/// Directly pushes LustInt to the specialized vector
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_vec_int_push(
    vec_ptr: *mut *mut LustInt,
    vec_len: *mut usize,
    vec_cap: *mut usize,
    value: LustInt,
) -> u8 {
    if vec_ptr.is_null() || vec_len.is_null() || vec_cap.is_null() {
        return 0;
    }

    let ptr = *vec_ptr;
    let len = *vec_len;
    let cap = *vec_cap;

    // Reconstruct Vec temporarily
    let mut vec = Vec::from_raw_parts(ptr, len, cap);

    // Push the value
    vec.push(value);

    // Extract new metadata
    let new_len = vec.len();
    let new_cap = vec.capacity();
    let new_ptr = vec.as_mut_ptr();

    // Prevent drop
    core::mem::forget(vec);

    // Update outputs
    ptr::write(vec_ptr, new_ptr);
    ptr::write(vec_len, new_len);
    ptr::write(vec_cap, new_cap);

    1
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_enum_is_some_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8 {
    if enum_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }

    let enum_value = &*enum_ptr;
    match enum_value {
        Value::Enum { variant, .. } => {
            let is_some = variant == "Some";
            ptr::write(out_ptr, Value::Bool(is_some));
            1
        }
        _ => 0,
    }
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_enum_unwrap_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8 {
    if enum_ptr.is_null() || out_ptr.is_null() {
        return 0;
    }

    let enum_value = &*enum_ptr;
    match enum_value {
        Value::Enum {
            values: Some(vals), ..
        } if vals.len() == 1 => {
            ptr::write(out_ptr, vals[0].clone());
            1
        }
        _ => 0,
    }
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_set_field_strong_safe(
    object_ptr: *const Value,
    field_index: usize,
    value_ptr: *const Value,
) -> u8 {
    if object_ptr.is_null() || value_ptr.is_null() {
        return 0;
    }

    let object = &*object_ptr;
    let value = (&*value_ptr).clone();

    match object {
        Value::Struct { fields, .. } => {
            // Skip canonicalization for strong fields - just set directly
            match fields.try_borrow_mut() {
                Ok(mut borrowed) => {
                    if field_index < borrowed.len() {
                        borrowed[field_index] = value;
                        1
                    } else {
                        0
                    }
                }
                Err(_) => 0,
            }
        }
        _ => 0,
    }
}

#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn jit_concat_safe(
    left_value_ptr: *const Value,
    right_value_ptr: *const Value,
    out: *mut Value,
) -> u8 {
    if left_value_ptr.is_null() || right_value_ptr.is_null() || out.is_null() {
        return 0;
    }

    let left = &*left_value_ptr;
    let right = &*right_value_ptr;
    const NO_VM_ERROR: &str = "task API requires a running VM";
    let left_str = match VM::with_current(|vm| {
        let left_copy = left.clone();
        vm.value_to_string_for_concat(&left_copy)
            .map_err(|err| err.to_string())
    }) {
        Ok(rc) => rc,
        Err(err) if err == NO_VM_ERROR => Rc::new(left.to_string()),
        Err(_) => return 0,
    };
    let right_str = match VM::with_current(|vm| {
        let right_copy = right.clone();
        vm.value_to_string_for_concat(&right_copy)
            .map_err(|err| err.to_string())
    }) {
        Ok(rc) => rc,
        Err(err) if err == NO_VM_ERROR => Rc::new(right.to_string()),
        Err(_) => return 0,
    };
    let mut combined = String::with_capacity(left_str.len() + right_str.len());
    combined.push_str(left_str.as_ref());
    combined.push_str(right_str.as_ref());
    let result = Value::string(combined);
    ptr::write(out, result);
    1
}

#[no_mangle]
pub unsafe extern "C" fn jit_guard_native_function(
    value_ptr: *const Value,
    expected_fn_ptr: *const (),
    register_index: u8,
) -> u8 {
    if value_ptr.is_null() || expected_fn_ptr.is_null() {
        jit::log(|| "jit_guard_native_function: null pointer input".to_string());
        return 0;
    }

    match &*value_ptr {
        Value::NativeFunction(func) => {
            let actual = Rc::as_ptr(func) as *const ();
            if actual == expected_fn_ptr {
                1
            } else {
                jit::log(|| {
                    format!(
                        "jit_guard_native_function: pointer mismatch (reg {}) actual={:p} expected={:p}",
                        register_index, actual, expected_fn_ptr
                    )
                });
                0
            }
        }

        other => {
            jit::log(|| {
                format!(
                    "jit_guard_native_function: value not native in reg {} ({:?})",
                    register_index,
                    other.tag()
                )
            });
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_guard_function_identity(
    value_ptr: *const Value,
    expected_kind: u8,
    expected_function_idx: usize,
    expected_upvalues: *const (),
    register_index: u8,
) -> u8 {
    if value_ptr.is_null() {
        jit::log(|| "jit_guard_function_identity: null pointer input".to_string());
        return 0;
    }

    let value = &*value_ptr;
    match (expected_kind, value) {
        (0, Value::Function(idx)) => {
            if *idx == expected_function_idx {
                1
            } else {
                jit::log(|| {
                    format!(
                        "jit_guard_function_identity: function idx mismatch (reg {}) actual={} expected={}",
                        register_index, idx, expected_function_idx
                    )
                });
                0
            }
        }

        (
            1,
            Value::Closure {
                function_idx,
                upvalues,
            },
        ) => {
            if *function_idx != expected_function_idx {
                jit::log(|| {
                    format!(
                        "jit_guard_function_identity: closure idx mismatch (reg {}) actual={} expected={}",
                        register_index, function_idx, expected_function_idx
                    )
                });
                return 0;
            }

            let actual_ptr = Rc::as_ptr(upvalues) as *const ();
            if actual_ptr == expected_upvalues {
                1
            } else {
                jit::log(|| {
                    format!(
                        "jit_guard_function_identity: upvalues mismatch (reg {}) actual={:p} expected={:p}",
                        register_index, actual_ptr, expected_upvalues
                    )
                });
                0
            }
        }

        (0, Value::Closure { function_idx, .. }) => {
            jit::log(|| {
                format!(
                    "jit_guard_function_identity: expected function, saw closure (reg {}, idx {})",
                    register_index, function_idx
                )
            });
            0
        }

        (1, Value::Function(idx)) => {
            jit::log(|| {
                format!(
                    "jit_guard_function_identity: expected closure, saw function (reg {}, idx {})",
                    register_index, idx
                )
            });
            0
        }

        (_, other) => {
            jit::log(|| {
                format!(
                    "jit_guard_function_identity: value in reg {} not callable ({:?})",
                    register_index,
                    other.tag()
                )
            });
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_call_native_safe(
    vm_ptr: *mut VM,
    callee_ptr: *const Value,
    expected_fn_ptr: *const (),
    args_ptr: *const Value,
    arg_count: u8,
    out: *mut Value,
) -> u8 {
    if vm_ptr.is_null() || callee_ptr.is_null() || expected_fn_ptr.is_null() || out.is_null() {
        jit::log(|| "jit_call_native_safe: null argument".to_string());
        return 0;
    }

    let callee = &*callee_ptr;
    let native_fn = match callee {
        Value::NativeFunction(func) => func.clone(),
        other => {
            jit::log(|| {
                format!(
                    "jit_call_native_safe: callee not native ({:?})",
                    other.tag()
                )
            });
            return 0;
        }
    };

    if Rc::as_ptr(&native_fn) as *const () != expected_fn_ptr {
        jit::log(|| {
            format!(
                "jit_call_native_safe: pointer mismatch actual={:p} expected={:p}",
                Rc::as_ptr(&native_fn),
                expected_fn_ptr
            )
        });
        return 0;
    }

    let mut args = Vec::with_capacity(arg_count as usize);
    if arg_count > 0 {
        if args_ptr.is_null() {
            jit::log(|| "jit_call_native_safe: args_ptr null with non-zero arg_count".to_string());
            return 0;
        }

        for i in 0..(arg_count as usize) {
            let arg = &*args_ptr.add(i);
            args.push(arg.clone());
        }
    }

    push_vm_ptr(vm_ptr);
    let outcome = native_fn(&args);
    pop_vm_ptr();

    let outcome = match outcome {
        Ok(result) => result,
        Err(err) => {
            jit::log(|| format!("jit_call_native_safe: native returned error: {}", err));
            return 0;
        }
    };

    match outcome {
        NativeCallResult::Return(value) => {
            ptr::write(out, value);
            1
        }

        NativeCallResult::Yield(_) => {
            jit::log(|| "jit_call_native_safe: native attempted to yield".to_string());
            0
        }

        NativeCallResult::Stop(_) => {
            jit::log(|| "jit_call_native_safe: native attempted to stop".to_string());
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_call_function_safe(
    vm_ptr: *mut VM,
    callee_ptr: *const Value,
    args_ptr: *const Value,
    arg_count: u8,
    dest_reg: u8,
) -> u8 {
    if vm_ptr.is_null() || callee_ptr.is_null() {
        jit::log(|| "jit_call_function_safe: null argument".to_string());
        return 0;
    }

    if arg_count > 0 && args_ptr.is_null() {
        jit::log(|| "jit_call_function_safe: args_ptr null with non-zero arg_count".to_string());
        return 0;
    }

    // Clone the callee BEFORE any operations that might reallocate registers
    let callee = (&*callee_ptr).clone();
    let mut args = Vec::with_capacity(arg_count as usize);
    for i in 0..(arg_count as usize) {
        let arg_ptr = args_ptr.add(i);
        args.push((&*arg_ptr).clone());
    }

    let vm = &mut *vm_ptr;
    push_vm_ptr(vm_ptr);

    // Temporarily disable JIT to prevent recursive JIT execution
    let jit_was_enabled = vm.jit.enabled;
    vm.jit.enabled = false;

    let call_result = vm.call_value(&callee, args);

    // Restore JIT state
    vm.jit.enabled = jit_was_enabled;
    pop_vm_ptr();

    match call_result {
        Ok(value) => {
            // Get current registers pointer AFTER the call (it may have reallocated)
            let vm = &mut *vm_ptr;
            if let Some(frame) = vm.call_stack.last_mut() {
                if (dest_reg as usize) < frame.registers.len() {
                    frame.registers[dest_reg as usize] = value;
                    1
                } else {
                    jit::log(|| {
                        format!(
                            "jit_call_function_safe: dest_reg {} out of bounds",
                            dest_reg
                        )
                    });
                    0
                }
            } else {
                jit::log(|| "jit_call_function_safe: no call frame".to_string());
                0
            }
        }

        Err(err) => {
            jit::log(|| format!("jit_call_function_safe: {}", err));
            0
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_current_registers(vm_ptr: *mut VM) -> *mut Value {
    if vm_ptr.is_null() {
        return core::ptr::null_mut();
    }

    let vm = &mut *vm_ptr;
    vm.call_stack
        .last_mut()
        .map(|frame| frame.registers.as_mut_ptr())
        .unwrap_or(core::ptr::null_mut())
}

#[no_mangle]
pub unsafe extern "C" fn jit_value_is_truthy(value_ptr: *const Value) -> u8 {
    if value_ptr.is_null() {
        return 0;
    }

    let value = &*value_ptr;
    if value.is_truthy() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_new_enum_unit_safe(
    enum_name_ptr: *const u8,
    enum_name_len: usize,
    variant_name_ptr: *const u8,
    variant_name_len: usize,
    out: *mut Value,
) -> u8 {
    if enum_name_ptr.is_null() || variant_name_ptr.is_null() || out.is_null() {
        return 0;
    }

    let enum_name_slice = slice::from_raw_parts(enum_name_ptr, enum_name_len);
    let variant_name_slice = slice::from_raw_parts(variant_name_ptr, variant_name_len);
    let enum_name = match str::from_utf8(enum_name_slice) {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let variant_name = match str::from_utf8(variant_name_slice) {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let value = Value::enum_unit(enum_name, variant_name);
    ptr::write(out, value);
    1
}

#[no_mangle]
pub unsafe extern "C" fn jit_new_enum_variant_safe(
    enum_name_ptr: *const u8,
    enum_name_len: usize,
    variant_name_ptr: *const u8,
    variant_name_len: usize,
    values_ptr: *const Value,
    value_count: usize,
    out: *mut Value,
) -> u8 {
    if enum_name_ptr.is_null() || variant_name_ptr.is_null() || out.is_null() {
        return 0;
    }

    if value_count > 0 && values_ptr.is_null() {
        return 0;
    }

    let enum_name_slice = slice::from_raw_parts(enum_name_ptr, enum_name_len);
    let variant_name_slice = slice::from_raw_parts(variant_name_ptr, variant_name_len);
    let enum_name = match str::from_utf8(enum_name_slice) {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let variant_name = match str::from_utf8(variant_name_slice) {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let mut values = Vec::with_capacity(value_count);
    for i in 0..value_count {
        let value = &*values_ptr.add(i);
        values.push(value.clone());
    }

    let value = Value::enum_variant(enum_name, variant_name, values);
    ptr::write(out, value);
    1
}

#[no_mangle]
pub unsafe extern "C" fn jit_is_enum_variant_safe(
    value_ptr: *const Value,
    enum_name_ptr: *const u8,
    enum_name_len: usize,
    variant_name_ptr: *const u8,
    variant_name_len: usize,
) -> u8 {
    if value_ptr.is_null() || enum_name_ptr.is_null() || variant_name_ptr.is_null() {
        return 0;
    }

    let value = &*value_ptr;
    let enum_name_slice = slice::from_raw_parts(enum_name_ptr, enum_name_len);
    let variant_name_slice = slice::from_raw_parts(variant_name_ptr, variant_name_len);
    let enum_name = match str::from_utf8(enum_name_slice) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let variant_name = match str::from_utf8(variant_name_slice) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    if value.is_enum_variant(enum_name, variant_name) {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_get_enum_value_safe(
    enum_ptr: *const Value,
    index: usize,
    out: *mut Value,
) -> u8 {
    if enum_ptr.is_null() || out.is_null() {
        return 0;
    }

    let enum_value = &*enum_ptr;
    if let Some((_, _, Some(values))) = enum_value.as_enum() {
        if index < values.len() {
            ptr::write(out, values[index].clone());
            1
        } else {
            0
        }
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_call_method_safe(
    vm_ptr: *mut VM,
    object_ptr: *const Value,
    method_name_ptr: *const u8,
    method_name_len: usize,
    args_ptr: *const Value,
    arg_count: u8,
    dest_reg: u8,
) -> u8 {
    if vm_ptr.is_null() || object_ptr.is_null() || method_name_ptr.is_null() {
        jit::log(|| "jit_call_method_safe: null pointer argument".to_string());
        return 0;
    }

    if arg_count > 0 && args_ptr.is_null() {
        return 0;
    }

    let method_name_slice = slice::from_raw_parts(method_name_ptr, method_name_len);
    let method_name = match str::from_utf8(method_name_slice) {
        Ok(s) => s,
        Err(_) => return 0,
    };

    let object = (&*object_ptr).clone();
    if matches!(object, Value::Struct { .. }) {
        return 0;
    }

    let mut args = Vec::with_capacity(arg_count as usize);
    for i in 0..arg_count {
        let arg_ptr = args_ptr.add(i as usize);
        args.push((&*arg_ptr).clone());
    }

    crate::vm::push_vm_ptr(vm_ptr);
    let outcome = call_builtin_method_simple(&object, method_name, args);
    crate::vm::pop_vm_ptr();
    match outcome {
        Ok(val) => {
            let vm = &mut *vm_ptr;
            if let Some(frame) = vm.call_stack.last_mut() {
                if (dest_reg as usize) < frame.registers.len() {
                    frame.registers[dest_reg as usize] = val;
                    1
                } else {
                    jit::log(|| {
                        format!("jit_call_method_safe: dest_reg {} out of bounds", dest_reg)
                    });
                    0
                }
            } else {
                jit::log(|| "jit_call_method_safe: no call frame".to_string());
                0
            }
        }
        Err(_) => 0,
    }
}

fn call_builtin_method_simple(
    object: &Value,
    method_name: &str,
    args: Vec<Value>,
) -> Result<Value, String> {
    match object {
        Value::Struct { name, .. } => Err(format!(
            "User-defined methods on {} require deoptimization",
            name
        )),
        Value::Iterator(state_rc) => match method_name {
            "next" => {
                let mut state = state_rc.borrow_mut();
                match &mut *state {
                    IteratorState::Array { items, index } => {
                        if *index < items.len() {
                            let v = items[*index].clone();
                            *index += 1;
                            Ok(Value::some(v))
                        } else {
                            Ok(Value::none())
                        }
                    }

                    IteratorState::MapPairs { items, index } => {
                        if *index < items.len() {
                            let (k, v) = items[*index].clone();
                            *index += 1;
                            Ok(Value::some(Value::array(vec![k.to_value(), v])))
                        } else {
                            Ok(Value::none())
                        }
                    }
                }
            }

            _ => Err(format!(
                "Iterator method '{}' not supported in JIT",
                method_name
            )),
        },
        Value::Enum {
            enum_name,
            variant,
            values,
        } if enum_name == "Option" => match method_name {
            "is_some" => Ok(Value::Bool(variant == "Some")),
            "is_none" => Ok(Value::Bool(variant == "None")),
            "unwrap" => {
                if variant == "Some" {
                    if let Some(vals) = values {
                        if vals.len() == 1 {
                            Ok(vals[0].clone())
                        } else {
                            Err("Option::Some should have exactly 1 value".to_string())
                        }
                    } else {
                        Err("Option::Some should have a value".to_string())
                    }
                } else {
                    Err("Called unwrap() on Option::None".to_string())
                }
            }

            _ => Err(format!(
                "Option method '{}' not supported in JIT",
                method_name
            )),
        },
        Value::Array(arr) => match method_name {
            "len" => Ok(Value::Int(int_from_usize(arr.borrow().len()))),
            "push" => {
                let value = args
                    .get(0)
                    .cloned()
                    .ok_or_else(|| "Array:push requires a value argument".to_string())?;
                arr.borrow_mut().push(value);
                Ok(Value::Nil)
            }
            "pop" => {
                let popped = arr.borrow_mut().pop();
                Ok(popped.map(Value::some).unwrap_or_else(Value::none))
            }
            "first" => {
                let borrowed = arr.borrow();
                Ok(borrowed
                    .first()
                    .cloned()
                    .map(Value::some)
                    .unwrap_or_else(Value::none))
            }
            "last" => {
                let borrowed = arr.borrow();
                Ok(borrowed
                    .last()
                    .cloned()
                    .map(Value::some)
                    .unwrap_or_else(Value::none))
            }
            "get" => {
                let index = args
                    .get(0)
                    .and_then(Value::as_int)
                    .ok_or_else(|| "Array:get requires an integer index".to_string())?;
                let borrowed = arr.borrow();
                Ok(borrowed
                    .get(index as usize)
                    .cloned()
                    .map(Value::some)
                    .unwrap_or_else(Value::none))
            }
            "iter" => {
                let items = arr.borrow().clone();
                let iter = IteratorState::Array { items, index: 0 };
                Ok(Value::Iterator(Rc::new(RefCell::new(iter))))
            }
            _ => Err(format!(
                "Array method '{}' not supported in JIT",
                method_name
            )),
        },
        _ => Err(format!(
            "Method '{}' not supported in JIT (deoptimizing)",
            method_name
        )),
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_get_field_safe(
    object_ptr: *const Value,
    field_name_ptr: *const u8,
    field_name_len: usize,
    out: *mut Value,
) -> u8 {
    if object_ptr.is_null() || field_name_ptr.is_null() || out.is_null() {
        return 0;
    }

    let field_name_slice = slice::from_raw_parts(field_name_ptr, field_name_len);
    let field_name = match str::from_utf8(field_name_slice) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let object = &*object_ptr;
    let field_value = match object {
        Value::Struct { layout, fields, .. } => match layout.index_of_str(field_name) {
            Some(idx) => match fields.borrow().get(idx) {
                Some(val) => val.clone(),
                None => return 0,
            },
            None => return 0,
        },
        _ => return 0,
    };
    ptr::write(out, field_value);
    1
}

#[no_mangle]
pub unsafe extern "C" fn jit_set_field_safe(
    object_ptr: *const Value,
    field_name_ptr: *const u8,
    field_name_len: usize,
    value_ptr: *const Value,
) -> u8 {
    if object_ptr.is_null() || field_name_ptr.is_null() || value_ptr.is_null() {
        return 0;
    }

    let field_name_slice = slice::from_raw_parts(field_name_ptr, field_name_len);
    let field_name = match str::from_utf8(field_name_slice) {
        Ok(s) => s,
        Err(_) => return 0,
    };
    let object = &*object_ptr;
    let value = (&*value_ptr).clone();
    match object {
        Value::Struct { .. } => match object.struct_set_field(field_name, value) {
            Ok(()) => 1,
            Err(_) => 0,
        },
        Value::Map(map) => {
            use crate::bytecode::ValueKey;
            let key = ValueKey::String(Rc::new(field_name.to_string()));
            map.borrow_mut().insert(key, value);
            1
        }

        _ => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_get_field_indexed_safe(
    object_ptr: *const Value,
    field_index: usize,
    out: *mut Value,
) -> u8 {
    if object_ptr.is_null() || out.is_null() {
        return 0;
    }

    let object = &*object_ptr;
    match object.struct_get_field_indexed(field_index) {
        Some(value) => {
            ptr::write(out, value);
            1
        }

        None => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_set_field_indexed_safe(
    object_ptr: *const Value,
    field_index: usize,
    value_ptr: *const Value,
) -> u8 {
    if object_ptr.is_null() || value_ptr.is_null() {
        return 0;
    }

    let object = &*object_ptr;
    let value = (&*value_ptr).clone();
    match object.struct_set_field_indexed(field_index, value) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_get_field_indexed_int_fast(
    object_ptr: *const Value,
    field_index: usize,
    out: *mut Value,
) -> u8 {
    if object_ptr.is_null() || out.is_null() {
        return 0;
    }

    let object = &*object_ptr;
    let out_ref = &mut *out;
    match object {
        Value::Struct { layout, fields, .. } => {
            if layout.is_weak(field_index) {
                return 0;
            }

            if let Ok(borrowed) = fields.try_borrow() {
                if let Some(Value::Int(val)) = borrowed.get(field_index) {
                    *out_ref = Value::Int(*val);
                    return 1;
                }
            }

            0
        }

        _ => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_set_field_indexed_int_fast(
    object_ptr: *const Value,
    field_index: usize,
    value_ptr: *const Value,
) -> u8 {
    if object_ptr.is_null() || value_ptr.is_null() {
        return 0;
    }

    let object = &*object_ptr;
    let value = &*value_ptr;
    let new_value = match value {
        Value::Int(v) => *v,
        _ => return 0,
    };
    match object {
        Value::Struct { layout, fields, .. } => {
            if layout.is_weak(field_index) {
                return 0;
            }

            if let Ok(mut borrowed) = fields.try_borrow_mut() {
                if field_index < borrowed.len() {
                    borrowed[field_index] = Value::Int(new_value);
                    return 1;
                }
            }

            0
        }

        _ => 0,
    }
}

#[no_mangle]
pub unsafe extern "C" fn jit_new_struct_safe(
    vm_ptr: *mut VM,
    struct_name_ptr: *const u8,
    struct_name_len: usize,
    field_names_ptr: *const *const u8,
    field_name_lens_ptr: *const usize,
    field_values_ptr: *const Value,
    field_count: usize,
    out: *mut Value,
) -> u8 {
    if struct_name_ptr.is_null() || out.is_null() || vm_ptr.is_null() {
        jit::log(|| "jit_new_struct_safe: null pointer input".to_string());
        return 0;
    }

    if field_count > 0
        && (field_names_ptr.is_null()
            || field_name_lens_ptr.is_null()
            || field_values_ptr.is_null())
    {
        return 0;
    }

    let struct_name_slice = slice::from_raw_parts(struct_name_ptr, struct_name_len);
    let struct_name = match str::from_utf8(struct_name_slice) {
        Ok(s) => s.to_string(),
        Err(_) => return 0,
    };
    let mut fields = Vec::with_capacity(field_count);
    for i in 0..field_count {
        let field_name_ptr = *field_names_ptr.add(i);
        let field_name_len = *field_name_lens_ptr.add(i);
        let field_name_slice = slice::from_raw_parts(field_name_ptr, field_name_len);
        let field_name = match str::from_utf8(field_name_slice) {
            Ok(s) => Rc::new(s.to_string()),
            Err(_) => return 0,
        };
        let field_value_ptr = field_values_ptr.add(i);
        let field_value = (&*field_value_ptr).clone();
        fields.push((field_name, field_value));
    }

    let vm = &mut *vm_ptr;
    let struct_value = match vm.instantiate_struct(&struct_name, fields) {
        Ok(value) => value,
        Err(err) => {
            jit::log(|| {
                format!(
                    "jit_new_struct_safe: failed to instantiate '{}': {}",
                    struct_name, err
                )
            });
            return 0;
        }
    };
    ptr::write(out, struct_value);
    1
}

#[no_mangle]
pub unsafe extern "C" fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8 {
    if src_ptr.is_null() || dest_ptr.is_null() {
        return 0;
    }

    let src_value = &*src_ptr;
    let cloned_value = src_value.clone();
    ptr::write(dest_ptr, cloned_value);
    1
}
