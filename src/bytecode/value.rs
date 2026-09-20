use crate::ast::Type;
use crate::jit;
use crate::number::{
    LustFloat, LustInt, float_from_int, float_is_nan, float_to_hash_bits, int_from_float,
    int_from_usize,
};
use crate::vm::{VM, pop_vm_ptr, push_vm_ptr};
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
// FixedState (not hashbrown's DefaultHashBuilder/foldhash RandomState): RandomState
// resolves its global seed from a per-crate-copy static, so maps created by the host
// binary are unreadable by extension cdylibs that statically link their own copy.
use foldhash::fast::FixedState as DefaultHashBuilder;
use hashbrown::HashMap;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskHandle(pub u64);
impl TaskHandle {
    pub fn id(&self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct ValueKey {
    original: Value,
    hashed: Value,
}

impl ValueKey {
    pub fn from_value(value: &Value) -> Self {
        ValueKey {
            original: value.clone(),
            hashed: value.clone(),
        }
    }

    pub fn with_hashed(original: Value, hashed: Value) -> Self {
        ValueKey { original, hashed }
    }

    pub fn string<S>(value: S) -> Self
    where
        S: Into<String>,
    {
        ValueKey::from(Value::String(Rc::new(value.into())))
    }

    pub fn to_value(&self) -> Value {
        self.original.clone()
    }

    pub(crate) fn owned_values(&self) -> (&Value, &Value) {
        (&self.original, &self.hashed)
    }
}

impl PartialEq for ValueKey {
    fn eq(&self, other: &Self) -> bool {
        value_key_eq(&self.hashed, &other.hashed)
    }
}

impl Eq for ValueKey {}
impl Hash for ValueKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_value_for_key(&self.hashed, state);
    }
}

impl fmt::Display for ValueKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.original)
    }
}

impl From<Value> for ValueKey {
    fn from(value: Value) -> Self {
        ValueKey {
            original: value.clone(),
            hashed: value,
        }
    }
}

impl From<LustInt> for ValueKey {
    fn from(value: LustInt) -> Self {
        ValueKey::from(Value::Int(value))
    }
}

impl From<LustFloat> for ValueKey {
    fn from(value: LustFloat) -> Self {
        ValueKey::from(Value::Float(value))
    }
}

impl From<bool> for ValueKey {
    fn from(value: bool) -> Self {
        ValueKey::from(Value::Bool(value))
    }
}

impl From<String> for ValueKey {
    fn from(value: String) -> Self {
        ValueKey::from(Value::String(Rc::new(value)))
    }
}

impl From<&str> for ValueKey {
    fn from(value: &str) -> Self {
        ValueKey::from(Value::String(Rc::new(value.to_owned())))
    }
}

impl From<Rc<String>> for ValueKey {
    fn from(value: Rc<String>) -> Self {
        ValueKey::from(Value::String(value))
    }
}

/// Helper function to unwrap LuaValue enums for key comparison
fn unwrap_lua_value_for_key(value: &Value) -> Value {
    if let Value::Enum(object) = value
        && object.enum_name == "LuaValue"
    {
        let values = &object.values;
        return match object.variant.as_str() {
            "Nil" => Value::Nil,
            "Bool" => values
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or(Value::Bool(false)),
            "Int" | "Number" => values
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or(Value::Int(0)),
            "String" => values
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or(Value::String(Rc::new(String::new()))),
            "Table" => values
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or(Value::Nil),
            "Function" => values
                .as_ref()
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or(Value::Nil),
            _ => value.clone(),
        };
    }
    value.clone()
}

fn value_key_eq(left: &Value, right: &Value) -> bool {
    use Value::*;

    // Check if either side is a LuaValue enum and unwrap if needed
    let is_lua_value_left = matches!(left, Enum(object) if object.enum_name == "LuaValue");
    let is_lua_value_right = matches!(right, Enum(object) if object.enum_name == "LuaValue");

    // If either side is a LuaValue, unwrap both and compare
    if is_lua_value_left || is_lua_value_right {
        let unwrapped_left = unwrap_lua_value_for_key(left);
        let unwrapped_right = unwrap_lua_value_for_key(right);
        return value_key_eq(&unwrapped_left, &unwrapped_right);
    }

    match (left, right) {
        (Nil, Nil) => true,
        (Bool(a), Bool(b)) => a == b,
        (Int(a), Int(b)) => a == b,
        (Float(a), Float(b)) => {
            if float_is_nan(*a) && float_is_nan(*b) {
                true
            } else {
                a == b
            }
        }
        (String(a), String(b)) => a == b,
        (Array(a), Array(b)) => Rc::ptr_eq(a, b),
        (Tuple(a), Tuple(b)) => Rc::ptr_eq(a, b),
        (Map(a), Map(b)) => Rc::ptr_eq(a, b),
        (Struct(a), Struct(b)) => Rc::ptr_eq(a, b),
        (WeakStruct(a), WeakStruct(b)) => Weak::ptr_eq(a.inner(), b.inner()),
        // Two enum values are the same value or two unit values of the
        // same variant; a payload is identity, not contents.
        (Enum(a), Enum(b)) => {
            Rc::ptr_eq(a, b)
                || (a.enum_name == b.enum_name
                    && a.variant == b.variant
                    && a.values.is_none()
                    && b.values.is_none())
        }
        (Function(a), Function(b)) => a == b,
        (NativeFunction(a), NativeFunction(b)) => Rc::ptr_eq(a, b),
        (Closure(a), Closure(b)) => Rc::ptr_eq(a, b),
        (Iterator(a), Iterator(b)) => Rc::ptr_eq(a, b),
        (Task(a), Task(b)) => a == b,
        _ => false,
    }
}

fn hash_value_for_key<H: Hasher>(value: &Value, state: &mut H) {
    use Value::*;

    // Check if this is a LuaValue enum and unwrap it for consistent hashing
    if matches!(value, Enum(object) if object.enum_name == "LuaValue") {
        let unwrapped = unwrap_lua_value_for_key(value);
        return hash_value_for_key(&unwrapped, state);
    }

    match value {
        Nil => {
            0u8.hash(state);
        }
        Bool(b) => {
            1u8.hash(state);
            b.hash(state);
        }
        Int(i) => {
            2u8.hash(state);
            i.hash(state);
        }
        Float(f) => {
            3u8.hash(state);
            if float_is_nan(*f) {
                u64::MAX.hash(state);
            } else {
                float_to_hash_bits(*f).hash(state);
            }
        }
        String(s) => {
            4u8.hash(state);
            s.hash(state);
        }
        Array(arr) => {
            5u8.hash(state);
            (Rc::as_ptr(arr) as usize).hash(state);
        }
        Tuple(tuple) => {
            6u8.hash(state);
            (Rc::as_ptr(tuple) as usize).hash(state);
        }
        Map(map) => {
            7u8.hash(state);
            (Rc::as_ptr(map) as usize).hash(state);
        }
        Struct(object) => {
            8u8.hash(state);
            (Rc::as_ptr(object) as usize).hash(state);
        }
        WeakStruct(weak) => {
            9u8.hash(state);
            (weak.object_ptr() as usize).hash(state);
        }
        Enum(object) => {
            10u8.hash(state);
            object.enum_name.hash(state);
            object.variant.hash(state);
            object
                .values
                .as_ref()
                .map(|_| Rc::as_ptr(object) as usize)
                .hash(state);
        }
        Function(idx) => {
            11u8.hash(state);
            idx.hash(state);
        }
        NativeFunction(func) => {
            12u8.hash(state);
            (Rc::as_ptr(func) as *const () as usize).hash(state);
        }
        Closure(closure) => {
            13u8.hash(state);
            (Rc::as_ptr(closure) as usize).hash(state);
        }
        Iterator(iter) => {
            14u8.hash(state);
            (Rc::as_ptr(iter) as usize).hash(state);
        }
        Task(handle) => {
            15u8.hash(state);
            handle.hash(state);
        }
    }
}

pub type LustMap = HashMap<ValueKey, Value, DefaultHashBuilder>;

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
            Value::Enum(object) if object.enum_name == "Option" => {
                let variant = &object.variant;
                let values = &object.values;
                if variant == "Some" {
                    if let Some(inner_values) = values {
                        if let Some(inner) = inner_values.first() {
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
            Value::Enum(object) if object.enum_name == "Option" => {
                let variant = &object.variant;
                let values = &object.values;
                if variant == "Some" {
                    if let Some(inner_values) = values {
                        if let Some(inner) = inner_values.first() {
                            match inner {
                                Value::WeakStruct(weak) => {
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
            Value::Struct(object) => Ok(Value::WeakStruct(WeakStructRef::new(&object))),
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

/// A struct or enum type name carried by a value: shared, so cloning the
/// value (every register move does) is a reference-count bump instead of
/// a string allocation. Compares and formats like the `str` it holds.
///
/// Names are interned per thread (values never cross threads): two names
/// with the same text are the same allocation, so equality is a pointer
/// comparison, and generated code compares a value's variant against a
/// constant the same way.
#[derive(Clone, Eq, Hash, PartialOrd, Ord)]
pub struct Name(Rc<str>);

#[cfg(feature = "std")]
thread_local! {
    static NAMES: RefCell<hashbrown::HashSet<Rc<str>>> = RefCell::new(hashbrown::HashSet::new());
}

impl Name {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(feature = "std")]
    fn intern(text: &str) -> Self {
        NAMES.with(|names| {
            let mut names = names.borrow_mut();
            if let Some(existing) = names.get(text) {
                return Name(Rc::clone(existing));
            }
            let fresh: Rc<str> = Rc::from(text);
            names.insert(Rc::clone(&fresh));
            Name(fresh)
        })
    }

    #[cfg(not(feature = "std"))]
    fn intern(text: &str) -> Self {
        Name(Rc::from(text))
    }

    /// The address generated code sees in a value holding this name: the
    /// shared allocation's start (the `Rc` stores that, not the text).
    pub fn inner_ptr(&self) -> *const u8 {
        // SAFETY: an `Rc<str>` allocation is `RcInner { strong, weak, text }`
        // (two counts, then the data); the pointer is only compared, never
        // dereferenced.
        (Rc::as_ptr(&self.0) as *const u8).wrapping_sub(2 * core::mem::size_of::<usize>())
    }
}

impl PartialEq for Name {
    fn eq(&self, other: &Name) -> bool {
        Rc::ptr_eq(&self.0, &other.0) || (cfg!(not(feature = "std")) && self.0 == other.0)
    }
}

impl core::ops::Deref for Name {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for Name {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl core::borrow::Borrow<str> for Name {
    fn borrow(&self) -> &str {
        &self.0
    }
}

impl PartialEq<str> for Name {
    fn eq(&self, other: &str) -> bool {
        &*self.0 == other
    }
}

impl PartialEq<&str> for Name {
    fn eq(&self, other: &&str) -> bool {
        &*self.0 == *other
    }
}

impl PartialEq<String> for Name {
    fn eq(&self, other: &String) -> bool {
        &*self.0 == other.as_str()
    }
}

impl PartialEq<Name> for str {
    fn eq(&self, other: &Name) -> bool {
        self == &*other.0
    }
}

impl PartialEq<Name> for String {
    fn eq(&self, other: &Name) -> bool {
        self.as_str() == &*other.0
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl fmt::Debug for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.0, f)
    }
}

impl From<String> for Name {
    fn from(s: String) -> Self {
        Name::intern(&s)
    }
}

impl From<&String> for Name {
    fn from(s: &String) -> Self {
        Name::intern(s)
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Name::intern(s)
    }
}

impl From<Name> for String {
    fn from(name: Name) -> Self {
        name.0.to_string()
    }
}

// Every heap variant is one thin `Rc`, so a `Value` is a tag and an 8-byte
// payload: the JIT lays registers out at that stride and copies values as
// one 16-byte pair, and everything that holds nothing owned is a scalar,
// a function index or a task handle.
const _: () = assert!(core::mem::size_of::<Value>() == 16);

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
    Map(Rc<RefCell<LustMap>>),
    Struct(Rc<StructObject>),
    WeakStruct(WeakStructRef),
    Enum(Rc<EnumObject>),
    Function(usize),
    NativeFunction(Rc<NativeFn>),
    Closure(Rc<ClosureObject>),
    Iterator(Rc<RefCell<IteratorState>>),
    Task(TaskHandle),
}

/// A struct value: its name, its layout and its fields, in one allocation
/// that every clone of the value shares.
#[repr(C)]
pub struct StructObject {
    pub name: Name,
    pub layout: Rc<StructLayout>,
    pub fields: RefCell<Vec<Value>>,
}

impl StructObject {
    pub fn new(name: impl Into<Name>, layout: Rc<StructLayout>, fields: Vec<Value>) -> Rc<Self> {
        Rc::new(Self {
            name: name.into(),
            layout,
            fields: RefCell::new(fields),
        })
    }
}

impl fmt::Debug for StructObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Struct {{ name: {:?}, fields: {:?} }}",
            self.name.as_str(),
            self.fields.borrow()
        )
    }
}

/// An enum value: the enum's name, the variant's, and the variant's payload
/// (`None` for a unit variant).
#[repr(C)]
pub struct EnumObject {
    pub enum_name: Name,
    pub variant: Name,
    pub values: Option<Vec<Value>>,
}

impl EnumObject {
    pub fn new(
        enum_name: impl Into<Name>,
        variant: impl Into<Name>,
        values: Option<Vec<Value>>,
    ) -> Rc<Self> {
        Rc::new(Self {
            enum_name: enum_name.into(),
            variant: variant.into(),
            values,
        })
    }
}

impl fmt::Debug for EnumObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Enum {{ enum: {:?}, variant: {:?}, values: {:?} }}",
            self.enum_name.as_str(),
            self.variant.as_str(),
            self.values
        )
    }
}

/// A closure: the function and the upvalues it captured.
#[repr(C)]
pub struct ClosureObject {
    pub function_idx: usize,
    pub upvalues: Vec<Upvalue>,
}

impl fmt::Debug for ClosureObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Closure {{ function: {}, upvalues: {:?} }}",
            self.function_idx, self.upvalues
        )
    }
}

/// Wrap a native function for a `Value::NativeFunction`.
pub fn native_fn<F>(f: F) -> Rc<NativeFn>
where
    F: Fn(&[Value]) -> Result<NativeCallResult, String> + 'static,
{
    Rc::new(Rc::new(f))
}

/// A weak reference to a struct value (a `weak` field): upgrades to the
/// struct while some strong reference keeps it alive.
#[derive(Debug, Clone)]
pub struct WeakStructRef(Weak<StructObject>);

impl WeakStructRef {
    pub fn new(object: &Rc<StructObject>) -> Self {
        Self(Rc::downgrade(object))
    }

    pub fn upgrade(&self) -> Option<Value> {
        self.0.upgrade().map(Value::Struct)
    }

    /// The struct's name, while it is alive.
    pub fn struct_name(&self) -> Option<Name> {
        self.0.upgrade().map(|object| object.name.clone())
    }

    pub(crate) fn inner(&self) -> &Weak<StructObject> {
        &self.0
    }

    pub(crate) fn object_ptr(&self) -> *const StructObject {
        self.0.as_ptr()
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

    pub(crate) fn cell(&self) -> &Rc<RefCell<Value>> {
        &self.value
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
            Value::Enum(_) => ValueTag::Enum,
            Value::Function(_) => ValueTag::Function,
            Value::NativeFunction(_) => ValueTag::NativeFunction,
            Value::Closure(_) => ValueTag::Closure,
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
            Value::Enum(_) => ValueType::Enum,
            Value::Function(_) => ValueType::Function,
            Value::NativeFunction(_) => ValueType::NativeFunction,
            Value::Closure(_) => ValueType::Closure,
            Value::Iterator(_) => ValueType::Iterator,
            Value::Task(_) => ValueType::Task,
        }
    }

    /// A value with no owned payload: copying its bits is a valid clone and
    /// overwriting it needs no drop. The interpreter's hot paths use this to
    /// skip the general `Clone`/`Drop` for scalars.
    ///
    /// `Function` is a plain index into the function table. `NativeFunction`
    /// is deliberately *not* here: it holds an `Rc`, so a bit copy would not
    /// bump the count and an overwrite would not release it (the JIT's own
    /// ownership probe lists it in `single_rc_tags` for the same reason).
    #[inline]
    pub fn is_plain(&self) -> bool {
        matches!(
            self,
            Value::Nil | Value::Bool(_) | Value::Int(_) | Value::Float(_) | Value::Function(_)
        )
    }

    /// `clone` with the scalar case inlined.
    #[inline]
    pub fn fast_clone(&self) -> Value {
        if self.is_plain() {
            // SAFETY: plain variants own nothing, so a bit copy is an
            // independent value.
            unsafe { core::ptr::read(self) }
        } else {
            self.clone()
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(false) => false,
            Value::Enum(object) if object.enum_name == "Option" && object.variant == "None" => {
                false
            }
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

    pub fn as_map(&self) -> Option<LustMap> {
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

    pub fn map(entries: LustMap) -> Self {
        Value::Map(Rc::new(RefCell::new(entries)))
    }

    pub fn task(handle: TaskHandle) -> Self {
        Value::Task(handle)
    }

    pub fn struct_get_field_rc(&self, field: &Rc<String>) -> Option<Value> {
        match self {
            Value::Struct(object) => object
                .layout
                .index_of_rc(field)
                .or_else(|| object.layout.index_of_str(field.as_str()))
                .or_else(|| {
                    object
                        .layout
                        .field_names()
                        .iter()
                        .position(|name| name.as_str() == field.as_str())
                })
                .and_then(|idx| {
                    object
                        .fields
                        .borrow()
                        .get(idx)
                        .cloned()
                        .map(|value| object.layout.materialize_field_value(idx, value))
                }),
            _ => None,
        }
    }

    pub fn struct_get_field(&self, field: &str) -> Option<Value> {
        match self {
            Value::Struct(object) => object.layout.index_of_str(field).and_then(|idx| {
                object
                    .fields
                    .borrow()
                    .get(idx)
                    .cloned()
                    .map(|value| object.layout.materialize_field_value(idx, value))
            }),
            _ => None,
        }
    }

    pub fn struct_get_field_indexed(&self, index: usize) -> Option<Value> {
        match self {
            Value::Struct(object) => object
                .fields
                .borrow()
                .get(index)
                .cloned()
                .map(|value| object.layout.materialize_field_value(index, value)),
            _ => None,
        }
    }

    pub fn struct_set_field_rc(&self, field: &Rc<String>, value: Value) -> Result<(), String> {
        match self {
            Value::Struct(object) => {
                let layout = &object.layout;
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
            Value::Struct(object) => {
                let layout = &object.layout;
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
            Value::Struct(object) => {
                let mut borrowed = object.fields.borrow_mut();
                if index < borrowed.len() {
                    let canonical = object.layout.canonicalize_field_value(index, value)?;
                    borrowed[index] = canonical;
                    Ok(())
                } else {
                    Err(format!(
                        "Struct '{}' field index {} out of bounds (len {})",
                        object.name,
                        index,
                        borrowed.len()
                    ))
                }
            }

            _ => Err("Attempted to set field on non-struct value".to_string()),
        }
    }

    pub fn enum_unit(enum_name: impl Into<Name>, variant: impl Into<Name>) -> Self {
        Value::Enum(EnumObject::new(enum_name, variant, None))
    }

    pub fn enum_variant(
        enum_name: impl Into<Name>,
        variant: impl Into<Name>,
        values: Vec<Value>,
    ) -> Self {
        Value::Enum(EnumObject::new(enum_name, variant, Some(values)))
    }

    pub fn as_enum(&self) -> Option<(&str, &str, Option<&[Value]>)> {
        match self {
            Value::Enum(object) => Some((
                object.enum_name.as_str(),
                object.variant.as_str(),
                object.values.as_deref(),
            )),
            _ => None,
        }
    }

    pub fn is_enum_variant(&self, enum_name: &str, variant: &str) -> bool {
        match self {
            Value::Enum(object) => {
                (enum_name.is_empty() || object.enum_name == enum_name) && object.variant == variant
            }
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

    #[allow(clippy::inherent_to_string_shadow_display, clippy::inherent_to_string)]
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
            Value::Struct(object) => {
                let StructObject {
                    name,
                    layout,
                    fields,
                } = object.as_ref();
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

            Value::Enum(object) => {
                let EnumObject {
                    enum_name,
                    variant,
                    values,
                } = object.as_ref();
                write!(
                    f,
                    "Enum {{ enum: {:?}, variant: {:?}, values: {:?} }}",
                    enum_name, variant, values
                )
            }

            Value::Function(idx) => write!(f, "Function({})", idx),
            Value::NativeFunction(_) => write!(f, "NativeFunction(<fn>)"),
            Value::Closure(closure) => {
                let ClosureObject {
                    function_idx,
                    upvalues,
                } = closure.as_ref();
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

            Value::Struct(object) => {
                let StructObject {
                    name,
                    layout,
                    fields,
                } = object.as_ref();
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

            Value::Enum(object) => {
                let EnumObject {
                    enum_name,
                    variant,
                    values,
                } = object.as_ref();
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
            Value::Closure(closure) => write!(f, "<closure@{}>", closure.function_idx),
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
            (Value::Struct(a), Value::Struct(b)) => {
                let (n1, l1, f1) = (&a.name, &a.layout, &a.fields);
                let (n2, l2, f2) = (&b.name, &b.layout, &b.fields);
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
            (Value::Enum(a), Value::Enum(b)) => {
                a.enum_name == b.enum_name && a.variant == b.variant && a.values == b.values
            }
            (Value::Function(a), Value::Function(b)) => a == b,
            (Value::Closure(a), Value::Closure(b)) => Rc::ptr_eq(a, b),
            (Value::Iterator(_), Value::Iterator(_)) => false,
            (Value::Task(a), Value::Task(b)) => a == b,
            _ => false,
        }
    }
}

#[inline]
unsafe fn replace_value(dest: *mut Value, value: Value) {
    unsafe {
        drop(ptr::replace(dest, value));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_replace_int(dest: *mut Value, value: LustInt) -> u8 {
    unsafe {
        if dest.is_null() {
            return 0;
        }
        replace_value(dest, Value::Int(value));
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_replace_int32(dest: *mut Value, value: i32) -> u8 {
    unsafe { jit_replace_int(dest, value as LustInt) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_replace_float_bits(dest: *mut Value, bits: u64) -> u8 {
    unsafe {
        if dest.is_null() {
            return 0;
        }
        #[cfg(feature = "std")]
        let value = LustFloat::from_bits(bits);
        #[cfg(not(feature = "std"))]
        let value = LustFloat::from_bits(bits as u32);
        replace_value(dest, Value::Float(value));
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_replace_float32_bits(dest: *mut Value, bits: u32) -> u8 {
    unsafe {
        if dest.is_null() {
            return 0;
        }
        replace_value(dest, Value::Float(f32::from_bits(bits) as LustFloat));
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_replace_bool(dest: *mut Value, value: u8) -> u8 {
    unsafe {
        if dest.is_null() {
            return 0;
        }
        replace_value(dest, Value::Bool(value != 0));
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_replace_nil(dest: *mut Value) -> u8 {
    unsafe {
        if dest.is_null() {
            return 0;
        }
        replace_value(dest, Value::Nil);
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_init_nil(dest: *mut Value) -> u8 {
    unsafe {
        if dest.is_null() {
            return 0;
        }
        ptr::write(dest, Value::Nil);
        1
    }
}

/// A compiled function's `Return`: move `src` (null = Nil) into `dest`,
/// dropping what `dest` held and leaving `src` Nil so the frame's drop
/// pass does not see the value twice.
///
/// # Safety
/// `dest` points at a live `Value`; `src` is null or points at one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_return_value(src: *mut Value, dest: *mut Value) {
    unsafe {
        let value = if src.is_null() {
            Value::Nil
        } else {
            let value = ptr::read(src);
            ptr::write(src, Value::Nil);
            value
        };
        replace_value(dest, value);
    }
}

/// Drop a native frame's registers except the aliased ones (`mask`, bit
/// i = register i), which are bitwise copies of the caller's values and
/// own nothing.
///
/// # Safety
/// `values` points at `len` initialized values; the masked ones are never
/// read again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_drop_values_masked(values: *mut Value, len: usize, mask: u64) {
    unsafe {
        if values.is_null() {
            return;
        }
        for i in 0..len {
            if i < 64 && mask & (1 << i) != 0 {
                continue;
            }
            ptr::drop_in_place(values.add(i));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_drop_values(values: *mut Value, len: usize) {
    unsafe {
        if !values.is_null() && len != 0 {
            ptr::drop_in_place(core::ptr::slice_from_raw_parts_mut(values, len));
        }
    }
}

#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_array_get_safe(
    vm_ptr: *mut VM,
    array_value_ptr: *const Value,
    index_value_ptr: *const Value,
    out: *mut Value,
) -> u8 {
    unsafe {
        if vm_ptr.is_null()
            || array_value_ptr.is_null()
            || index_value_ptr.is_null()
            || out.is_null()
        {
            eprintln!("❌ jit_array_get_safe: null pointer detected!");
            return 0;
        }
        let Some(index) = (&*index_value_ptr).as_int() else {
            return 0;
        };

        let array_value = &*array_value_ptr;
        let arr = match array_value {
            Value::Array(arr) => arr,
            _ => {
                return 0;
            }
        };
        let borrowed = match arr.try_borrow() {
            Ok(b) => b,
            Err(_) => {
                return 0;
            }
        };
        let length = borrowed.len();
        if index < 0 || index as usize >= length {
            drop(borrowed);
            (&mut *vm_ptr).set_pending_jit_error(crate::LustError::RuntimeError {
                message: format!("Array index {} out of bounds (length: {})", index, length),
            });
            return 0;
        }

        let value = borrowed[index as usize].clone();
        drop(borrowed);
        replace_value(out, value);
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_array_index_result_safe(
    vm_ptr: *mut VM,
    array_value_ptr: *const Value,
    index_value_ptr: *const Value,
    out: *mut Value,
) -> u8 {
    unsafe {
        if vm_ptr.is_null()
            || array_value_ptr.is_null()
            || index_value_ptr.is_null()
            || out.is_null()
        {
            return 0;
        }
        let Some(index) = (&*index_value_ptr).as_int() else {
            return 0;
        };
        let vm = &mut *vm_ptr;
        let result = match vm.array_index_result(&*array_value_ptr, index) {
            Ok(result) => result,
            Err(error) => {
                vm.set_pending_jit_error(error);
                return 0;
            }
        };
        vm.observe_value(&result);
        replace_value(out, result);
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_array_index_ok_safe(
    array_value_ptr: *const Value,
    index_value_ptr: *const Value,
    value_out: *mut Value,
    condition_out: *mut Value,
) -> u8 {
    unsafe {
        if array_value_ptr.is_null()
            || index_value_ptr.is_null()
            || value_out.is_null()
            || condition_out.is_null()
        {
            return 0;
        }
        let Value::Array(array) = &*array_value_ptr else {
            return 0;
        };
        let Some(index) = (&*index_value_ptr).as_int() else {
            return 0;
        };
        let borrowed = match array.try_borrow() {
            Ok(borrowed) => borrowed,
            Err(_) => return 0,
        };
        let value = if index >= 0 {
            borrowed.get(index as usize).cloned()
        } else {
            None
        };
        drop(borrowed);

        if let Some(value) = value {
            replace_value(value_out, value);
            replace_value(condition_out, Value::Bool(true));
        } else {
            replace_value(value_out, Value::Nil);
            replace_value(condition_out, Value::Bool(false));
        }
        1
    }
}

#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_array_len_safe(array_value_ptr: *const Value) -> i64 {
    unsafe {
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
}

#[cfg(feature = "std")]
static JIT_NEW_ARRAY_COUNTER: core::sync::atomic::AtomicUsize =
    core::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_new_array_safe(
    vm_ptr: *mut VM,
    elements_ptr: *const Value,
    element_count: usize,
    out_ptr: *mut Value,
) -> u8 {
    unsafe {
        let _call_num = JIT_NEW_ARRAY_COUNTER.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        // jit::log(|| format!("jit_new_array_safe call #{}: ENTER - elements_ptr={:?}, count={}, out_ptr={:?}", call_num, elements_ptr, element_count, out_ptr));

        if out_ptr.is_null() {
            // jit::log(|| "jit_new_array_safe: out_ptr is null".to_string());
            return 0;
        }

        if !vm_ptr.is_null() {
            let vm = &mut *vm_ptr;
            if !vm.try_charge_memory_value_vec(element_count) {
                return 0;
            }
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
        if !vm_ptr.is_null() {
            (&mut *vm_ptr).observe_value(&array_value);
        }
        // jit::log(|| format!("jit_new_array_safe #{}: about to write to out_ptr", call_num));
        replace_value(out_ptr, array_value);
        // jit::log(|| format!("jit_new_array_safe #{}: EXIT - success, returning 1", call_num));
        1
    }
}

#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_array_push_safe(
    vm_ptr: *mut VM,
    array_ptr: *const Value,
    value_ptr: *const Value,
) -> u8 {
    unsafe {
        if array_ptr.is_null() || value_ptr.is_null() {
            return 0;
        }

        let array_value = &*array_ptr;
        let value = &*value_ptr;

        match array_value {
            Value::Array(arr) => {
                // Use unchecked borrow for maximum performance
                let cell_ptr = arr.as_ptr();
                let vec_ref = &mut *cell_ptr;
                if !vm_ptr.is_null() {
                    let vm = &mut *vm_ptr;
                    let len = vec_ref.len();
                    let cap = vec_ref.capacity();
                    if len == cap {
                        let new_cap = if cap == 0 { 4 } else { cap.saturating_mul(2) };
                        if !vm.try_charge_memory_vec_growth::<Value>(cap, new_cap) {
                            return 0;
                        }
                    }
                }
                vec_ref.push(value.clone());
                1
            }
            _ => 0,
        }
    }
}

/// A specialized `Array<int>` slot on the JIT stack: the raw parts of a
/// `Vec<LustInt>` copy of the elements, plus a strong reference to the
/// array it was copied from. Traces read and push through the copy; the
/// rebox publishes it back into that same array object, wherever the
/// register that held the array has moved on to by then. Zeroed = empty.
#[cfg(feature = "std")]
type ArrayRef = Rc<RefCell<Vec<Value>>>;

#[cfg(feature = "std")]
#[repr(C)]
pub struct JitVecSlot {
    ptr: *mut LustInt,
    len: usize,
    cap: usize,
    array: *const RefCell<Vec<Value>>,
}

#[cfg(feature = "std")]
impl JitVecSlot {
    /// Take the slot's contents, leaving it zeroed.
    ///
    /// # Safety
    /// The slot holds either zeros or what `jit_unbox_array_int` stored.
    unsafe fn take(&mut self) -> (Option<Vec<LustInt>>, Option<ArrayRef>) {
        let vec = if self.ptr.is_null() {
            None
        } else {
            Some(unsafe { Vec::from_raw_parts(self.ptr, self.len, self.cap) })
        };
        let array = if self.array.is_null() {
            None
        } else {
            Some(unsafe { Rc::from_raw(self.array) })
        };
        self.ptr = ptr::null_mut();
        self.len = 0;
        self.cap = 0;
        self.array = ptr::null();
        (vec, array)
    }
}

/// Unbox `Array<int>` into `slot`: copy the elements into a `Vec<LustInt>`
/// and keep a reference to the array. Whatever the slot held before (an
/// earlier unbox in an unrolled iteration) is released first; on failure
/// the slot is left empty. Returns 1 on success, 0 on failure.
///
/// # Safety
/// `array_value_ptr` points at a live `Value` and `slot` at a slot that
/// holds zeros or what an earlier call stored.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_unbox_array_int(
    array_value_ptr: *const Value,
    slot: *mut JitVecSlot,
) -> u8 {
    unsafe {
        if array_value_ptr.is_null() || slot.is_null() {
            return 0;
        }
        let slot = &mut *slot;
        drop(slot.take());

        let Value::Array(arr_rc) = &*array_value_ptr else {
            return 0;
        };
        // Copy the elements out; do NOT move them. Moving left the source
        // array empty for the span of the trace, and anything that re-entered
        // the runtime mid-trace (a nested `for elem in arr`) saw a hollowed-out
        // array. A runtime re-entry mid-trace can still see values that
        // predate pushes made by the trace; closing that hole needs escape
        // analysis so such arrays are never specialized at all.
        let vec_ref = &*arr_rc.as_ptr();
        if vec_ref.iter().any(|value| !matches!(value, Value::Int(_))) {
            return 0;
        }
        let mut specialized_vec: Vec<LustInt> = vec_ref
            .iter()
            .map(|value| match value {
                Value::Int(value) => *value,
                _ => unreachable!("array element types were validated above"),
            })
            .collect();
        slot.len = specialized_vec.len();
        slot.cap = specialized_vec.capacity();
        slot.ptr = specialized_vec.as_mut_ptr();
        core::mem::forget(specialized_vec);
        slot.array = Rc::into_raw(Rc::clone(arr_rc));
        1
    }
}

/// Rebox: write the slot's elements back into the array they were unboxed
/// from and empty the slot. An empty slot (nothing unboxed on this path) is
/// a no-op. Returns 1 on success, 0 on a malformed slot.
///
/// # Safety
/// `slot` points at a slot that holds zeros or what `jit_unbox_array_int`
/// stored.
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_rebox_array_int(slot: *mut JitVecSlot) -> u8 {
    unsafe {
        if slot.is_null() {
            return 0;
        }
        let (vec, array) = (*slot).take();
        match (vec, array) {
            (Some(vec), Some(array)) => {
                *array.as_ptr() = vec.into_iter().map(Value::Int).collect();
                1
            }
            (None, None) => 1,
            _ => 0,
        }
    }
}
/// Specialized push operation for Vec<LustInt>
/// Directly pushes LustInt to the specialized vector
#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_vec_int_push(
    vec_ptr: *mut *mut LustInt,
    vec_len: *mut usize,
    vec_cap: *mut usize,
    value: LustInt,
) -> u8 {
    unsafe {
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
}

/// Drop a specialized Vec<LustInt> (cleanup for leaked specializations)
/// WARNING: This should NOT be called! Specialized values that get invalidated
/// during loop recording don't actually exist on the stack during execution.
#[cfg(feature = "std")]
/// Drop a specialized slot without publishing it (its value was
/// invalidated during recording).
///
/// # Safety
/// `slot` points at a slot that holds zeros or what `jit_unbox_array_int`
/// stored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_drop_vec_int(slot: *mut JitVecSlot) {
    unsafe {
        if !slot.is_null() {
            drop((*slot).take());
        }
    }
}

#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_enum_is_some_safe(enum_ptr: *const Value, out_ptr: *mut Value) -> u8 {
    unsafe {
        if enum_ptr.is_null() || out_ptr.is_null() {
            return 0;
        }

        let enum_value = &*enum_ptr;
        match enum_value {
            Value::Enum(object) => {
                let EnumObject { variant, .. } = object.as_ref();
                let is_some = variant == "Some";
                replace_value(out_ptr, Value::Bool(is_some));
                1
            }
            _ => 0,
        }
    }
}

#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_enum_unwrap_safe(
    vm_ptr: *mut VM,
    enum_ptr: *const Value,
    out_ptr: *mut Value,
) -> u8 {
    unsafe {
        if vm_ptr.is_null() || enum_ptr.is_null() || out_ptr.is_null() {
            return 0;
        }

        let enum_value = &*enum_ptr;
        match enum_value {
            Value::Enum(object)
                if object.values.as_ref().is_some_and(|vals| vals.len() == 1)
                    && ((object.enum_name == "Option" && object.variant == "Some")
                        || (object.enum_name == "Result" && object.variant == "Ok")) =>
            {
                let value = object.values.as_ref().expect("checked")[0].clone();
                replace_value(out_ptr, value);
                1
            }
            Value::Enum(object) => {
                let EnumObject {
                    enum_name,
                    variant,
                    values,
                } = object.as_ref();
                let detail = values
                    .as_ref()
                    .and_then(|values| values.first())
                    .map(|value| format!(": {}", value))
                    .unwrap_or_default();
                (&mut *vm_ptr).set_pending_jit_error(crate::LustError::RuntimeError {
                    message: format!("Called unwrap() on {}::{}{}", enum_name, variant, detail),
                });
                0
            }
            value => {
                (&mut *vm_ptr).set_pending_jit_error(crate::LustError::RuntimeError {
                    message: format!("Cannot unwrap {:?}", value.type_of()),
                });
                0
            }
        }
    }
}

#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_set_field_strong_safe(
    object_ptr: *const Value,
    field_index: usize,
    value_ptr: *const Value,
) -> u8 {
    unsafe {
        if object_ptr.is_null() || value_ptr.is_null() {
            return 0;
        }

        let object = &*object_ptr;
        let value = (&*value_ptr).clone();

        match object {
            Value::Struct(object) => {
                let StructObject { fields, .. } = object.as_ref();
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
}

#[cfg(feature = "std")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_concat_safe(
    vm_ptr: *mut VM,
    left_value_ptr: *const Value,
    right_value_ptr: *const Value,
    out: *mut Value,
) -> u8 {
    unsafe {
        if left_value_ptr.is_null() || right_value_ptr.is_null() || out.is_null() {
            return 0;
        }

        let left = &*left_value_ptr;
        let right = &*right_value_ptr;
        let (left_str, right_str) = if !vm_ptr.is_null() {
            let vm = &mut *vm_ptr;
            let left_copy = left.clone();
            let right_copy = right.clone();
            let left_str = match vm.value_to_string_for_concat(&left_copy) {
                Ok(rc) => rc,
                Err(_) => return 0,
            };
            let right_str = match vm.value_to_string_for_concat(&right_copy) {
                Ok(rc) => rc,
                Err(_) => return 0,
            };
            (left_str, right_str)
        } else {
            (Rc::new(left.to_string()), Rc::new(right.to_string()))
        };
        let cap = left_str.len().saturating_add(right_str.len());
        if !vm_ptr.is_null() {
            let vm = &mut *vm_ptr;
            if !vm.try_charge_memory_bytes(cap) {
                return 0;
            }
        }
        let mut combined = String::with_capacity(cap);
        combined.push_str(left_str.as_ref());
        combined.push_str(right_str.as_ref());
        let result = Value::string(combined);
        replace_value(out, result);
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_guard_native_function(
    value_ptr: *const Value,
    expected_fn_ptr: *const (),
    register_index: u8,
) -> u8 {
    unsafe {
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
}

/// Does the value hold a struct whose layout is `expected`?
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_guard_struct_layout(value_ptr: *const Value, expected: *const ()) -> u8 {
    unsafe {
        if value_ptr.is_null() {
            return 0;
        }
        match &*value_ptr {
            Value::Struct(object) => u8::from(Rc::as_ptr(&object.layout) as *const () == expected),
            _ => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_guard_function_identity(
    value_ptr: *const Value,
    expected_kind: u8,
    expected_function_idx: usize,
    expected_upvalues: *const (),
    register_index: u8,
) -> u8 {
    unsafe {
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

            (1, Value::Closure(closure)) => {
                let function_idx = &closure.function_idx;
                if *function_idx != expected_function_idx {
                    jit::log(|| {
                        format!(
                            "jit_guard_function_identity: closure idx mismatch (reg {}) actual={} expected={}",
                            register_index, function_idx, expected_function_idx
                        )
                    });
                    return 0;
                }

                // The closure's identity: the object every clone shares.
                let actual_ptr = Rc::as_ptr(closure) as *const ();
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

            (0, Value::Closure(closure)) => {
                let function_idx = closure.function_idx;
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
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_call_native_safe(
    vm_ptr: *mut VM,
    callee_ptr: *const Value,
    expected_fn_ptr: *const (),
    args_ptr: *const Value,
    arg_count: u8,
    out: *mut Value,
) -> u8 {
    unsafe {
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
                jit::log(|| {
                    "jit_call_native_safe: args_ptr null with non-zero arg_count".to_string()
                });
                return 0;
            }

            for i in 0..(arg_count as usize) {
                let arg = &*args_ptr.add(i);
                args.push(arg.clone());
            }
        }

        // The caller's frame shows the call's line if the native fails.
        if let Some(ip) = (&mut *vm_ptr).take_call_ip()
            && let Some(frame) = (&mut *vm_ptr).call_stack.last_mut()
        {
            frame.ip = ip + 1;
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
                (&mut *vm_ptr).observe_value_graph(&value);
                replace_value(out, value);
                1
            }

            NativeCallResult::Yield(value) => {
                let vm = &mut *vm_ptr;
                if vm.current_task.is_none() {
                    jit::log(|| {
                        "jit_call_native_safe: native attempted to yield outside a task".to_string()
                    });
                    return 0;
                }

                let dest_reg = vm.call_stack.last().and_then(|frame| {
                    let base = frame.registers.as_ptr() as usize;
                    let out_ptr = out as usize;
                    let value_size = core::mem::size_of::<Value>();
                    let end = base + value_size * frame.registers.len();
                    if out_ptr < base || out_ptr >= end {
                        return None;
                    }
                    let offset = out_ptr - base;
                    if !offset.is_multiple_of(value_size) {
                        return None;
                    }
                    let reg = offset / value_size;
                    if reg > u8::MAX as usize {
                        return None;
                    }
                    Some(reg as u8)
                });
                let Some(dest_reg) = dest_reg else {
                    jit::log(|| {
                        "jit_call_native_safe: could not compute dest register for yield"
                            .to_string()
                    });
                    return 0;
                };

                replace_value(out, Value::Nil);
                vm.observe_value_graph(&value);
                vm.pending_task_signal = Some(crate::vm::TaskSignal::Yield {
                    dest: dest_reg,
                    value,
                });
                2
            }

            NativeCallResult::Stop(value) => {
                let vm = &mut *vm_ptr;
                if vm.current_task.is_none() {
                    jit::log(|| {
                        "jit_call_native_safe: native attempted to stop outside a task".to_string()
                    });
                    return 0;
                }

                replace_value(out, Value::Nil);
                vm.observe_value_graph(&value);
                vm.pending_task_signal = Some(crate::vm::TaskSignal::Stop { value });
                3
            }
        }
    }
}

#[unsafe(no_mangle)]
/// Call a Lust function from generated code. The result goes to `out` when
/// it is non-null (a register of an inlined frame on the JIT stack, whose
/// address is stable), otherwise to register `dest_reg` of the VM's current
/// frame, looked up after the call because the call may reallocate the
/// register file.
pub unsafe extern "C" fn jit_call_function_safe(
    vm_ptr: *mut VM,
    callee_ptr: *const Value,
    args_ptr: *const Value,
    arg_count: u8,
    dest_reg: u8,
    out: *mut Value,
) -> u8 {
    unsafe {
        if vm_ptr.is_null() || callee_ptr.is_null() {
            jit::log(|| "jit_call_function_safe: null argument".to_string());
            return 0;
        }

        if arg_count > 0 && args_ptr.is_null() {
            jit::log(|| {
                "jit_call_function_safe: args_ptr null with non-zero arg_count".to_string()
            });
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
        // The caller's frame shows the call's line in a stack trace.
        if let Some(ip) = vm.take_call_ip()
            && let Some(frame) = vm.call_stack.last_mut()
        {
            frame.ip = ip + 1;
        }
        push_vm_ptr(vm_ptr);

        // The callee runs with the JIT on: its loops get their own traces
        // and its body its own compiled code. Nothing in the trace that
        // made this call depends on the JIT being idle meanwhile — it holds
        // its own code alive and has written every register back.
        let call_result = vm.call_value(&callee, args);

        pop_vm_ptr();

        match call_result {
            Ok(value) => {
                vm.observe_value_graph(&value);
                if !out.is_null() {
                    *out = value;
                    return 1;
                }
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
                vm.set_pending_jit_error(err);
                0
            }
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_current_registers(vm_ptr: *mut VM) -> *mut Value {
    unsafe {
        if vm_ptr.is_null() {
            return core::ptr::null_mut();
        }

        let vm = &mut *vm_ptr;
        vm.call_stack
            .last_mut()
            .map(|frame| frame.registers.as_mut_ptr())
            .unwrap_or(core::ptr::null_mut())
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_value_is_truthy(value_ptr: *const Value) -> u8 {
    unsafe {
        if value_ptr.is_null() {
            return 0;
        }

        let value = &*value_ptr;
        if value.is_truthy() { 1 } else { 0 }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_new_enum_unit_safe(
    vm_ptr: *mut VM,
    enum_name_ptr: *const u8,
    enum_name_len: usize,
    variant_name_ptr: *const u8,
    variant_name_len: usize,
    out: *mut Value,
) -> u8 {
    unsafe {
        if enum_name_ptr.is_null() || variant_name_ptr.is_null() || out.is_null() {
            return 0;
        }

        let enum_name_slice = slice::from_raw_parts(enum_name_ptr, enum_name_len);
        let variant_name_slice = slice::from_raw_parts(variant_name_ptr, variant_name_len);
        let enum_name_str = match str::from_utf8(enum_name_slice) {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let variant_name_str = match str::from_utf8(variant_name_slice) {
            Ok(s) => s,
            Err(_) => return 0,
        };

        let value = if vm_ptr.is_null() {
            Value::enum_unit(enum_name_str, variant_name_str)
        } else {
            (*vm_ptr).unit_enum(enum_name_str, variant_name_str)
        };
        replace_value(out, value);
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_new_enum_variant_safe(
    vm_ptr: *mut VM,
    enum_name_ptr: *const u8,
    enum_name_len: usize,
    variant_name_ptr: *const u8,
    variant_name_len: usize,
    values_ptr: *const Value,
    value_count: usize,
    out: *mut Value,
) -> u8 {
    unsafe {
        if enum_name_ptr.is_null() || variant_name_ptr.is_null() || out.is_null() {
            return 0;
        }

        if value_count > 0 && values_ptr.is_null() {
            return 0;
        }

        let enum_name_slice = slice::from_raw_parts(enum_name_ptr, enum_name_len);
        let variant_name_slice = slice::from_raw_parts(variant_name_ptr, variant_name_len);
        let enum_name_str = match str::from_utf8(enum_name_slice) {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let variant_name_str = match str::from_utf8(variant_name_slice) {
            Ok(s) => s,
            Err(_) => return 0,
        };

        if !vm_ptr.is_null() {
            let vm = &mut *vm_ptr;
            let name_bytes = enum_name_len.saturating_add(variant_name_len);
            if !vm.try_charge_memory_bytes(name_bytes) {
                return 0;
            }
            if !vm.try_charge_memory_value_vec(value_count) {
                return 0;
            }
        }

        let enum_name = enum_name_str.to_string();
        let variant_name = variant_name_str.to_string();
        let mut values = Vec::with_capacity(value_count);
        for i in 0..value_count {
            let value = &*values_ptr.add(i);
            values.push(value.clone());
        }

        let value = Value::enum_variant(enum_name, variant_name, values);
        replace_value(out, value);
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_is_enum_variant_safe(
    value_ptr: *const Value,
    enum_name_ptr: *const u8,
    enum_name_len: usize,
    variant_name_ptr: *const u8,
    variant_name_len: usize,
) -> u8 {
    unsafe {
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
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_type_is_safe(
    vm_ptr: *mut VM,
    value_ptr: *const Value,
    type_name_ptr: *const u8,
    type_name_len: usize,
) -> u8 {
    unsafe {
        if vm_ptr.is_null() || value_ptr.is_null() || type_name_ptr.is_null() {
            return 0;
        }

        let type_name = match str::from_utf8(slice::from_raw_parts(type_name_ptr, type_name_len)) {
            Ok(name) => name,
            Err(_) => return 0,
        };
        if (&*vm_ptr).value_is_type(&*value_ptr, type_name) {
            1
        } else {
            0
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_try_cast_safe(
    vm_ptr: *mut VM,
    value_ptr: *const Value,
    type_name_ptr: *const u8,
    type_name_len: usize,
    out: *mut Value,
) -> u8 {
    unsafe {
        if vm_ptr.is_null() || value_ptr.is_null() || type_name_ptr.is_null() || out.is_null() {
            return 0;
        }

        let type_name = match str::from_utf8(slice::from_raw_parts(type_name_ptr, type_name_len)) {
            Ok(name) => name,
            Err(_) => return 0,
        };
        let vm = &mut *vm_ptr;
        let result = if vm.value_is_type(&*value_ptr, type_name) {
            if !vm.try_charge_memory_value_vec(1) {
                return 0;
            }
            Value::some((*value_ptr).clone())
        } else {
            Value::none()
        };
        vm.observe_value(&result);
        replace_value(out, result);
        1
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_get_enum_value_safe(
    enum_ptr: *const Value,
    index: usize,
    out: *mut Value,
) -> u8 {
    unsafe {
        if enum_ptr.is_null() || out.is_null() {
            return 0;
        }

        let enum_value = &*enum_ptr;
        if let Some((_, _, Some(values))) = enum_value.as_enum() {
            if index < values.len() {
                let value = values[index].clone();
                replace_value(out, value);
                1
            } else {
                0
            }
        } else {
            0
        }
    }
}

#[unsafe(no_mangle)]
/// Call a builtin method from generated code. Result delivery follows
/// `jit_call_function_safe`: `out` when non-null, else `dest_reg` of the
/// VM's current frame.
pub unsafe extern "C" fn jit_call_method_safe(
    vm_ptr: *mut VM,
    object_ptr: *const Value,
    method_name_ptr: *const u8,
    method_name_len: usize,
    args_ptr: *const Value,
    arg_count: u8,
    dest_reg: u8,
    out: *mut Value,
) -> u8 {
    unsafe {
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
                if !out.is_null() {
                    *out = val;
                    return 1;
                }
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
}

fn call_builtin_method_simple(
    object: &Value,
    method_name: &str,
    args: Vec<Value>,
) -> Result<Value, String> {
    match object {
        Value::Struct(object) => Err(format!(
            "User-defined methods on {} require deoptimization",
            object.name
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
        Value::Enum(object) if object.enum_name == "Option" => {
            match_option_method(object, method_name, args)
        }
        Value::Enum(object) if object.enum_name == "Result" => {
            match_result_method(object, method_name, args)
        }
        _ => Err(format!(
            "Method '{}' not supported in JIT (deoptimizing)",
            method_name
        )),
    }
}

fn match_option_method(
    object: &EnumObject,
    method_name: &str,
    _args: Vec<Value>,
) -> Result<Value, String> {
    let variant = &object.variant;
    let values = &object.values;
    match method_name {
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
    }
}

fn match_result_method(
    object: &EnumObject,
    method_name: &str,
    args: Vec<Value>,
) -> Result<Value, String> {
    let variant = &object.variant;
    let values = &object.values;
    match method_name {
        "is_ok" => Ok(Value::Bool(variant == "Ok")),
        "is_err" => Ok(Value::Bool(variant == "Err")),
        "unwrap" => {
            if variant == "Ok" {
                values
                    .as_ref()
                    .and_then(|values| values.first())
                    .cloned()
                    .ok_or_else(|| "Result::Ok should have exactly 1 value".to_string())
            } else {
                Err("Called unwrap() on Result::Err".to_string())
            }
        }
        "unwrap_or" => {
            let default = args
                .first()
                .cloned()
                .ok_or_else(|| "Result:unwrap_or requires a default value".to_string())?;
            if variant == "Ok" {
                Ok(values
                    .as_ref()
                    .and_then(|values| values.first())
                    .cloned()
                    .unwrap_or(default))
            } else {
                Ok(default)
            }
        }
        _ => Err(format!(
            "Result method '{}' not supported in JIT",
            method_name
        )),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_get_field_safe(
    object_ptr: *const Value,
    field_name_ptr: *const u8,
    field_name_len: usize,
    out: *mut Value,
) -> u8 {
    unsafe {
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
            Value::Struct(object) => match object.layout.index_of_str(field_name) {
                Some(idx) => match object.fields.borrow().get(idx) {
                    Some(val) => val.clone(),
                    None => return 0,
                },
                None => return 0,
            },
            _ => return 0,
        };
        replace_value(out, field_value);
        1
    }
}

/// `jit_get_field_safe` that also reads a `Map` (a module table, say),
/// keyed by `key`: the field name as a `ValueKey` the code retains, so
/// nothing is built per call. A missing entry reads as Nil, as in the
/// interpreter.
///
/// # Safety
/// `object_ptr` and `key` point to live values, `out` to a live value slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_get_field_keyed(
    object_ptr: *const Value,
    field_name_ptr: *const u8,
    field_name_len: usize,
    key: *const ValueKey,
    out: *mut Value,
) -> u8 {
    unsafe {
        if let Value::Map(map) = &*object_ptr {
            let value = match map.try_borrow() {
                Ok(map) => map.get(&*key).cloned().unwrap_or(Value::Nil),
                Err(_) => return 0,
            };
            replace_value(out, value);
            return 1;
        }
        jit_get_field_safe(object_ptr, field_name_ptr, field_name_len, out)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_set_field_safe(
    object_ptr: *const Value,
    field_name_ptr: *const u8,
    field_name_len: usize,
    value_ptr: *const Value,
) -> u8 {
    unsafe {
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
            Value::Struct(_) => match object.struct_set_field(field_name, value) {
                Ok(()) => 1,
                Err(_) => 0,
            },
            Value::Map(map) => {
                use crate::bytecode::ValueKey;
                let key = ValueKey::from(field_name.to_string());
                map.borrow_mut().insert(key, value);
                1
            }

            _ => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_get_field_indexed_safe(
    object_ptr: *const Value,
    field_index: usize,
    out: *mut Value,
) -> u8 {
    unsafe {
        if object_ptr.is_null() || out.is_null() {
            return 0;
        }

        let object = &*object_ptr;
        match object.struct_get_field_indexed(field_index) {
            Some(value) => {
                replace_value(out, value);
                1
            }

            None => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_set_field_indexed_safe(
    object_ptr: *const Value,
    field_index: usize,
    value_ptr: *const Value,
) -> u8 {
    unsafe {
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
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_get_field_indexed_int_fast(
    object_ptr: *const Value,
    field_index: usize,
    out: *mut Value,
) -> u8 {
    unsafe {
        if object_ptr.is_null() || out.is_null() {
            return 0;
        }

        let object = &*object_ptr;
        let out_ref = &mut *out;
        match object {
            Value::Struct(object) => {
                let StructObject { layout, fields, .. } = object.as_ref();
                if layout.is_weak(field_index) {
                    return 0;
                }

                if let Ok(borrowed) = fields.try_borrow()
                    && let Some(Value::Int(val)) = borrowed.get(field_index)
                {
                    *out_ref = Value::Int(*val);
                    return 1;
                }

                0
            }

            _ => 0,
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_set_field_indexed_int_fast(
    object_ptr: *const Value,
    field_index: usize,
    value_ptr: *const Value,
) -> u8 {
    unsafe {
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
            Value::Struct(object) => {
                let StructObject { layout, fields, .. } = object.as_ref();
                if layout.is_weak(field_index) {
                    return 0;
                }

                if let Ok(mut borrowed) = fields.try_borrow_mut()
                    && field_index < borrowed.len()
                {
                    borrowed[field_index] = Value::Int(new_value);
                    return 1;
                }

                0
            }

            _ => 0,
        }
    }
}

#[unsafe(no_mangle)]
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
    unsafe {
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
        vm.observe_value(&struct_value);
        replace_value(out, struct_value);
        1
    }
}

/// Take another reference to whatever the value owns (the counterpart of
/// a bitwise copy generated code made of it).
///
/// # Safety
/// `value` points at a live `Value`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_retain_value(value: *const Value) {
    unsafe {
        core::mem::forget((*value).clone());
    }
}

/// Drop the value in place, leaving Nil.
///
/// # Safety
/// `value` points at a live `Value` that generated code will overwrite or
/// no longer read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_release_value(value: *mut Value) {
    unsafe {
        replace_value(value, Value::Nil);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn jit_move_safe(src_ptr: *const Value, dest_ptr: *mut Value) -> u8 {
    unsafe {
        if src_ptr.is_null() || dest_ptr.is_null() {
            return 0;
        }

        let cloned_value = (&*src_ptr).clone();
        replace_value(dest_ptr, cloned_value);
        1
    }
}

#[cfg(test)]
mod value_ownership_tests {
    use super::*;

    /// `is_plain` decides whether the interpreter may copy a value's bits
    /// instead of cloning it (`fast_clone`) and overwrite a register
    /// instead of dropping it (`set_register`, `CallFrame::reset`). A
    /// variant holding an `Rc` must never qualify: the copy would not bump
    /// the count and the overwrite would not release it.
    #[test]
    fn native_functions_are_not_plain_values() {
        let native = native_fn(|_args: &[Value]| Ok(NativeCallResult::Return(Value::Nil)));
        let value = Value::NativeFunction(Rc::clone(&native));
        assert!(!value.is_plain());
        assert_eq!(Rc::strong_count(&native), 2);

        let copy = value.fast_clone();
        assert_eq!(Rc::strong_count(&native), 3);
        drop(copy);
        assert_eq!(Rc::strong_count(&native), 2);
    }

    /// The variants that may be bit-copied, spelled out: everything else
    /// owns a payload and goes through `Clone`/`Drop`.
    #[test]
    fn only_payload_free_variants_are_plain() {
        assert!(Value::Nil.is_plain());
        assert!(Value::Bool(true).is_plain());
        assert!(Value::Int(1).is_plain());
        assert!(Value::Float(1.0).is_plain());
        assert!(Value::Function(0).is_plain());

        assert!(!Value::string("owned").is_plain());
        assert!(!Value::array(alloc::vec![Value::Int(1)]).is_plain());
        assert!(!Value::some(Value::Int(1)).is_plain());
    }
}

#[cfg(test)]
mod jit_replacement_tests {
    use super::*;

    #[test]
    fn scalar_helpers_drop_owned_destinations() {
        let string = Rc::new("owned".to_string());
        let mut value = Value::String(string.clone());
        assert_eq!(Rc::strong_count(&string), 2);

        unsafe {
            assert_eq!(jit_replace_int(&mut value, 42), 1);
        }
        assert_eq!(Rc::strong_count(&string), 1);
        assert!(matches!(value, Value::Int(42)));

        let array = Rc::new(RefCell::new(vec![Value::Int(1)]));
        value = Value::Array(array.clone());
        assert_eq!(Rc::strong_count(&array), 2);
        unsafe {
            assert_eq!(jit_replace_bool(&mut value, 1), 1);
        }
        assert_eq!(Rc::strong_count(&array), 1);
        assert!(matches!(value, Value::Bool(true)));

        unsafe {
            assert_eq!(jit_replace_float_bits(&mut value, 1.5f64.to_bits()), 1);
        }
        assert!(matches!(value, Value::Float(float) if float == 1.5));
        unsafe {
            assert_eq!(jit_replace_nil(&mut value), 1);
        }
        assert!(matches!(value, Value::Nil));
    }

    #[test]
    fn move_helper_supports_self_move() {
        let string = Rc::new("self".to_string());
        let mut value = Value::String(string.clone());
        let value_ptr = &mut value as *mut Value;

        unsafe {
            assert_eq!(jit_move_safe(value_ptr, value_ptr), 1);
        }

        assert_eq!(Rc::strong_count(&string), 2);
        assert!(matches!(&value, Value::String(value) if Rc::ptr_eq(value, &string)));
    }

    #[cfg(feature = "std")]
    fn empty_slot() -> JitVecSlot {
        JitVecSlot {
            ptr: core::ptr::null_mut(),
            len: 0,
            cap: 0,
            array: core::ptr::null(),
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn failed_array_specialization_restores_original_values() {
        let value = Value::array(vec![Value::Int(1), Value::string("not an int")]);
        let mut slot = empty_slot();

        let result = unsafe { jit_unbox_array_int(&value, &mut slot) };

        assert_eq!(result, 0);
        assert!(slot.ptr.is_null() && slot.array.is_null());
        assert_eq!(unsafe { jit_rebox_array_int(&mut slot) }, 1);
        assert_eq!(value.array_len(), Some(2));
        assert!(matches!(value.array_get(0), Some(Value::Int(1))));
        assert!(matches!(value.array_get(1), Some(Value::String(_))));
    }

    /// The rebox publishes into the array that was unboxed, not into
    /// whatever register the trace last held it in: a guard exit taken while
    /// that register holds something else must leave the register alone.
    #[cfg(feature = "std")]
    #[test]
    fn rebox_publishes_into_the_unboxed_array() {
        let value = Value::array(vec![Value::Int(1), Value::Int(2)]);
        let Value::Array(rc) = &value else { unreachable!() };
        let mut slot = empty_slot();
        assert_eq!(unsafe { jit_unbox_array_int(&value, &mut slot) }, 1);
        assert_eq!(Rc::strong_count(rc), 2);

        // A push through the specialized copy, then a rebox.
        let mut vec = unsafe { Vec::from_raw_parts(slot.ptr, slot.len, slot.cap) };
        vec.push(3);
        slot.len = vec.len();
        slot.cap = vec.capacity();
        slot.ptr = vec.as_mut_ptr();
        core::mem::forget(vec);
        assert_eq!(unsafe { jit_rebox_array_int(&mut slot) }, 1);

        assert_eq!(value.array_len(), Some(3));
        assert!(matches!(value.array_get(2), Some(Value::Int(3))));
        assert_eq!(Rc::strong_count(rc), 1);
        assert!(slot.ptr.is_null() && slot.array.is_null());

        // Unboxing again over a slot that was never reboxed releases the old
        // copy and reference instead of leaking them.
        assert_eq!(unsafe { jit_unbox_array_int(&value, &mut slot) }, 1);
        assert_eq!(unsafe { jit_unbox_array_int(&value, &mut slot) }, 1);
        assert_eq!(Rc::strong_count(rc), 2);
        unsafe { jit_drop_vec_int(&mut slot) };
        assert_eq!(Rc::strong_count(rc), 1);
    }
}
