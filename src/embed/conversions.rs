use super::values::{
    ArrayHandle, EnumInstance, FunctionHandle, MapHandle, StringRef, StructHandle, StructInstance,
    TypedValue, ValueRef, matches_array_handle_type, matches_array_type,
    matches_function_handle_type, matches_lust_enum, matches_lust_struct, matches_map_handle_type,
};
use crate::ast::{Span, Type, TypeKind};
use crate::bytecode::Value;
use crate::number::{LustFloat, LustInt};
use crate::{LustError, Result};
use std::rc::Rc;

fn struct_field_type_error(field: &str, expected: &str, actual: &Value) -> LustError {
    LustError::TypeError {
        message: format!(
            "Struct field '{}' expects '{}' but received value of type '{:?}'",
            field,
            expected,
            actual.type_of()
        ),
    }
}

pub trait FromStructField<'a>: Sized {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self>;
}

impl<'a> FromStructField<'a> for ValueRef<'a> {
    fn from_value(_field: &str, value: ValueRef<'a>) -> Result<Self> {
        Ok(value)
    }
}

impl<'a> FromStructField<'a> for StructHandle {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_struct_handle()
            .ok_or_else(|| struct_field_type_error(field, "struct", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for StructInstance {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_struct_handle()
            .map(|handle| handle.to_instance())
            .ok_or_else(|| struct_field_type_error(field, "struct", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for ArrayHandle {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_array_handle()
            .ok_or_else(|| struct_field_type_error(field, "array", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for MapHandle {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_map_handle()
            .ok_or_else(|| struct_field_type_error(field, "map", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for FunctionHandle {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        let owned = value.to_owned();
        FunctionHandle::from_value(owned)
            .map_err(|_| struct_field_type_error(field, "function", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for LustInt {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_int()
            .ok_or_else(|| struct_field_type_error(field, "int", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for LustFloat {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_float()
            .ok_or_else(|| struct_field_type_error(field, "float", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for bool {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_bool()
            .ok_or_else(|| struct_field_type_error(field, "bool", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for Rc<String> {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        value
            .as_rc_string()
            .ok_or_else(|| struct_field_type_error(field, "string", value.as_value()))
    }
}

impl<'a> FromStructField<'a> for Value {
    fn from_value(_field: &str, value: ValueRef<'a>) -> Result<Self> {
        Ok(value.into_owned())
    }
}

impl<'a, T> FromStructField<'a> for Option<T>
where
    T: FromStructField<'a>,
{
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        if matches!(value.as_value(), Value::Nil) {
            Ok(None)
        } else {
            T::from_value(field, value).map(Some)
        }
    }
}

impl<'a> FromStructField<'a> for StringRef<'a> {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        if value.as_string().is_some() {
            Ok(StringRef::new(value))
        } else {
            Err(struct_field_type_error(field, "string", value.as_value()))
        }
    }
}

impl<'a> FromStructField<'a> for EnumInstance {
    fn from_value(field: &str, value: ValueRef<'a>) -> Result<Self> {
        match value.as_value() {
            Value::Enum(_) => <EnumInstance as FromLustValue>::from_value(value.into_owned()),
            other => Err(struct_field_type_error(field, "enum", other)),
        }
    }
}

pub trait LustStructView<'a>: Sized {
    const TYPE_NAME: &'static str;

    fn from_handle(handle: &'a StructHandle) -> Result<Self>;
}

pub trait IntoTypedValue {
    fn into_typed_value(self) -> TypedValue;
}

impl IntoTypedValue for Value {
    fn into_typed_value(self) -> TypedValue {
        TypedValue::new(self, |_value, _ty| true, "Value")
    }
}

impl IntoTypedValue for StructInstance {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, matches_lust_struct, "struct")
    }
}

impl IntoTypedValue for StructHandle {
    fn into_typed_value(self) -> TypedValue {
        <StructInstance as IntoTypedValue>::into_typed_value(self.into_instance())
    }
}

impl IntoTypedValue for EnumInstance {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, matches_lust_enum, "enum")
    }
}

impl IntoTypedValue for FunctionHandle {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, |_v, ty| matches_function_handle_type(ty), "function")
    }
}

macro_rules! impl_into_typed_for_primitive {
    ($ty:ty, $desc:expr, $matcher:expr) => {
        impl IntoTypedValue for $ty {
            fn into_typed_value(self) -> TypedValue {
                let value = self.into_value();
                TypedValue::new(value, $matcher, $desc)
            }
        }
    };
}

impl_into_typed_for_primitive!(LustInt, "int", |_, ty: &Type| match &ty.kind {
    TypeKind::Int | TypeKind::Unknown => true,
    TypeKind::Union(types) => types
        .iter()
        .any(|alt| matches!(&alt.kind, TypeKind::Int | TypeKind::Unknown)),
    _ => false,
});
impl_into_typed_for_primitive!(LustFloat, "float", |_, ty: &Type| match &ty.kind {
    TypeKind::Float | TypeKind::Unknown => true,
    TypeKind::Union(types) => types
        .iter()
        .any(|alt| matches!(&alt.kind, TypeKind::Float | TypeKind::Unknown)),
    _ => false,
});
impl_into_typed_for_primitive!(bool, "bool", |_, ty: &Type| match &ty.kind {
    TypeKind::Bool | TypeKind::Unknown => true,
    TypeKind::Union(types) => types
        .iter()
        .any(|alt| matches!(&alt.kind, TypeKind::Bool | TypeKind::Unknown)),
    _ => false,
});
impl IntoTypedValue for String {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, string_matcher, "string")
    }
}

impl IntoTypedValue for &str {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, string_matcher, "string")
    }
}

impl IntoTypedValue for &String {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, string_matcher, "string")
    }
}

impl IntoTypedValue for () {
    fn into_typed_value(self) -> TypedValue {
        TypedValue::new(
            Value::Nil,
            |_, ty| matches!(ty.kind, TypeKind::Unit | TypeKind::Unknown),
            "unit",
        )
    }
}

impl<T> IntoTypedValue for Vec<T>
where
    T: IntoLustValue,
{
    fn into_typed_value(self) -> TypedValue {
        let values = self.into_iter().map(|item| item.into_value()).collect();
        TypedValue::new(
            Value::array(values),
            |_, ty| matches_array_type(ty, &T::matches_lust_type),
            "array",
        )
    }
}

impl IntoTypedValue for ArrayHandle {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, |_, ty| matches_array_handle_type(ty), "array")
    }
}

impl IntoTypedValue for MapHandle {
    fn into_typed_value(self) -> TypedValue {
        let value = self.into_value();
        TypedValue::new(value, |_, ty| matches_map_handle_type(ty), "map")
    }
}

fn string_matcher(_: &Value, ty: &Type) -> bool {
    match &ty.kind {
        TypeKind::String | TypeKind::Unknown => true,
        TypeKind::Union(types) => types
            .iter()
            .any(|alt| matches!(&alt.kind, TypeKind::String | TypeKind::Unknown)),
        _ => false,
    }
}

pub trait FromLustArgs: Sized {
    fn from_values(values: &[Value]) -> std::result::Result<Self, String>;
    fn matches_signature(params: &[Type]) -> bool;
}

macro_rules! impl_from_lust_args_tuple {
    ($( $name:ident ),+) => {
        impl<$($name),+> FromLustArgs for ($($name,)+)
        where
            $($name: FromLustValue,)+
        {
            fn from_values(values: &[Value]) -> std::result::Result<Self, String> {
                let expected = count_idents!($($name),+);
                if values.len() != expected {
                    return Err(format!(
                        "Native function expected {} argument(s) but received {}",
                        expected,
                        values.len()
                    ));
                }

                let mut idx = 0;
                let result = (
                    $(
                        {
                            let value = $name::from_value(values[idx].clone()).map_err(|e| e.to_string())?;
                            idx += 1;
                            value
                        },
                    )+
                );
                let _ = idx;
                Ok(result)
            }

            fn matches_signature(params: &[Type]) -> bool {
                let expected = count_idents!($($name),+);
                params.len() == expected && {
                    let mut idx = 0;
                    let mut ok = true;
                    $(
                        if ok && !$name::matches_lust_type(&params[idx]) {
                            ok = false;
                        }

                        idx += 1;
                    )+
                    let _ = idx;
                    ok
                }

            }

        }

    };
}

macro_rules! count_idents {
    ($($name:ident),*) => {
        <[()]>::len(&[$(count_idents!(@sub $name)),*])
    };
    (@sub $name:ident) => { () };
}

impl_from_lust_args_tuple!(A);
impl_from_lust_args_tuple!(A, B);
impl_from_lust_args_tuple!(A, B, C);
impl_from_lust_args_tuple!(A, B, C, D);
impl_from_lust_args_tuple!(A, B, C, D, E);
impl<T> FromLustArgs for T
where
    T: FromLustValue,
{
    fn from_values(values: &[Value]) -> std::result::Result<Self, String> {
        match values.len() {
            0 => T::from_value(Value::Nil).map_err(|e| e.to_string()),
            1 => T::from_value(values[0].clone()).map_err(|e| e.to_string()),
            count => Err(format!(
                "Native function expected 1 argument but received {}",
                count
            )),
        }
    }

    fn matches_signature(params: &[Type]) -> bool {
        if params.is_empty() {
            let unit = Type::new(TypeKind::Unit, Span::new(0, 0, 0, 0));
            return T::matches_lust_type(&unit);
        }

        params.len() == 1 && T::matches_lust_type(&params[0])
    }
}

pub trait IntoLustValue: Sized {
    fn into_value(self) -> Value;
    fn matches_lust_type(ty: &Type) -> bool;
    fn type_description() -> &'static str;

    /// Whether this Rust value represents zero Lust arguments rather than one
    /// unit-valued argument. Defaults to `false` for custom conversions.
    fn is_unit() -> bool {
        false
    }
}

pub trait FromLustValue: Sized {
    fn from_value(value: Value) -> Result<Self>;
    fn matches_lust_type(ty: &Type) -> bool;
    fn type_description() -> &'static str;
}

pub trait FunctionArgs {
    /// Bundles a Rust value, unit, or a tuple of 2 through 12 values for a typed call.
    fn into_values(self) -> Vec<Value>;
    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()>;
}

impl IntoLustValue for Value {
    fn into_value(self) -> Value {
        self
    }

    fn matches_lust_type(_: &Type) -> bool {
        true
    }

    fn type_description() -> &'static str {
        "Value"
    }
}

impl FromLustValue for Value {
    fn from_value(value: Value) -> Result<Self> {
        Ok(value)
    }

    fn matches_lust_type(_: &Type) -> bool {
        true
    }

    fn type_description() -> &'static str {
        "Value"
    }
}

impl IntoLustValue for () {
    fn into_value(self) -> Value {
        Value::Nil
    }

    fn is_unit() -> bool {
        true
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Unit | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "unit"
    }
}

impl IntoLustValue for LustInt {
    fn into_value(self) -> Value {
        Value::Int(self)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Int | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "int"
    }
}

impl FromLustValue for LustInt {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Int(v) => Ok(v),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'int' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Int | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "int"
    }
}

impl IntoLustValue for LustFloat {
    fn into_value(self) -> Value {
        Value::Float(self)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Float | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "float"
    }
}

impl FromLustValue for LustFloat {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Float(v) => Ok(v),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'float' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Float | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "float"
    }
}

impl IntoLustValue for bool {
    fn into_value(self) -> Value {
        Value::Bool(self)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Bool | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "bool"
    }
}

impl FromLustValue for bool {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Bool(b) => Ok(b),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'bool' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Bool | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "bool"
    }
}

impl IntoLustValue for String {
    fn into_value(self) -> Value {
        Value::String(Rc::new(self))
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl IntoLustValue for Rc<String> {
    fn into_value(self) -> Value {
        Value::String(self)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl IntoLustValue for StructInstance {
    fn into_value(self) -> Value {
        self.into_value()
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown | TypeKind::Named(_) | TypeKind::GenericInstance { .. } => true,
            TypeKind::Union(types) => types.iter().any(<Self as IntoLustValue>::matches_lust_type),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "struct"
    }
}

impl IntoLustValue for StructHandle {
    fn into_value(self) -> Value {
        <StructInstance as IntoLustValue>::into_value(self.into_instance())
    }

    fn matches_lust_type(ty: &Type) -> bool {
        <StructInstance as IntoLustValue>::matches_lust_type(ty)
    }

    fn type_description() -> &'static str {
        <StructInstance as IntoLustValue>::type_description()
    }
}

impl IntoLustValue for FunctionHandle {
    fn into_value(self) -> Value {
        self.into_value()
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_function_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "function"
    }
}

impl FromLustValue for StructInstance {
    fn from_value(value: Value) -> Result<Self> {
        match &value {
            Value::Struct(object) => Ok(StructInstance::new(object.name.to_string(), value)),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'struct' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown | TypeKind::Named(_) | TypeKind::GenericInstance { .. } => true,
            TypeKind::Union(types) => types.iter().any(<Self as FromLustValue>::matches_lust_type),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "struct"
    }
}

impl FromLustValue for StructHandle {
    fn from_value(value: Value) -> Result<Self> {
        <StructInstance as FromLustValue>::from_value(value).map(StructHandle::from)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        <StructInstance as FromLustValue>::matches_lust_type(ty)
    }

    fn type_description() -> &'static str {
        <StructInstance as FromLustValue>::type_description()
    }
}

impl FromLustValue for FunctionHandle {
    fn from_value(value: Value) -> Result<Self> {
        if FunctionHandle::is_callable_value(&value) {
            Ok(FunctionHandle::new_unchecked(value))
        } else {
            Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'function' but received '{:?}'", value),
            })
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_function_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "function"
    }
}

impl IntoLustValue for EnumInstance {
    fn into_value(self) -> Value {
        self.into_value()
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown | TypeKind::Named(_) | TypeKind::GenericInstance { .. } => true,
            TypeKind::Union(types) => types.iter().any(<Self as IntoLustValue>::matches_lust_type),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "enum"
    }
}

impl FromLustValue for EnumInstance {
    fn from_value(value: Value) -> Result<Self> {
        match &value {
            Value::Enum(object) => Ok(EnumInstance::new(
                object.enum_name.to_string(),
                object.variant.to_string(),
                value,
            )),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'enum' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Unknown | TypeKind::Named(_) | TypeKind::GenericInstance { .. } => true,
            TypeKind::Union(types) => types.iter().any(<Self as FromLustValue>::matches_lust_type),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "enum"
    }
}

impl<T> IntoLustValue for Vec<T>
where
    T: IntoLustValue,
{
    fn into_value(self) -> Value {
        let values = self.into_iter().map(|item| item.into_value()).collect();
        Value::array(values)
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_array_type(ty, &T::matches_lust_type)
    }

    fn type_description() -> &'static str {
        "array"
    }
}

impl IntoLustValue for ArrayHandle {
    fn into_value(self) -> Value {
        self.into_value()
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_array_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "array"
    }
}

impl IntoLustValue for MapHandle {
    fn into_value(self) -> Value {
        self.into_value()
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_map_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "map"
    }
}

impl<T> FromLustValue for Vec<T>
where
    T: FromLustValue,
{
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Array(items) => {
                let borrowed = items.borrow();
                let mut result = Vec::with_capacity(borrowed.len());
                for item in borrowed.iter() {
                    result.push(T::from_value(item.clone())?);
                }

                Ok(result)
            }

            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'array' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_array_type(ty, &T::matches_lust_type)
    }

    fn type_description() -> &'static str {
        "array"
    }
}

impl FromLustValue for ArrayHandle {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Array(items) => Ok(ArrayHandle::from_rc(items)),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'array' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_array_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "array"
    }
}

impl FromLustValue for MapHandle {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Map(map) => Ok(MapHandle::from_rc(map)),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'map' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches_map_handle_type(ty)
    }

    fn type_description() -> &'static str {
        "map"
    }
}

impl IntoLustValue for &str {
    fn into_value(self) -> Value {
        Value::String(Rc::new(self.to_owned()))
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl IntoLustValue for &String {
    fn into_value(self) -> Value {
        Value::String(Rc::new(self.clone()))
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl FromLustValue for String {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::String(s) => Ok((*s).clone()),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'string' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl FromLustValue for Rc<String> {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::String(s) => Ok(s),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'string' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::String | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "string"
    }
}

impl FromLustValue for () {
    fn from_value(value: Value) -> Result<Self> {
        match value {
            Value::Nil => Ok(()),
            other => Err(LustError::RuntimeError {
                message: format!("Expected Lust value 'unit' but received '{:?}'", other),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        matches!(ty.kind, TypeKind::Unit | TypeKind::Unknown)
    }

    fn type_description() -> &'static str {
        "unit"
    }
}

impl<T> IntoLustValue for Option<T>
where
    T: IntoLustValue,
{
    fn into_value(self) -> Value {
        match self {
            Some(value) => Value::enum_variant("Option", "Some", vec![value.into_value()]),
            None => Value::enum_unit("Option", "None"),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Option(inner) => T::matches_lust_type(inner),
            TypeKind::Unknown => true,
            TypeKind::Union(types) => types.iter().any(Self::matches_lust_type),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "Option"
    }
}

impl<T> FromLustValue for Option<T>
where
    T: FromLustValue,
{
    fn from_value(value: Value) -> Result<Self> {
        let (variant, payload) = expected_enum_parts(&value, "Option")?;
        match (variant, payload) {
            ("None", None) => Ok(None),
            ("Some", Some([value])) => T::from_value(value.clone()).map(Some),
            ("None", Some(_)) => Err(LustError::RuntimeError {
                message: "Lust Option.None unexpectedly carries a payload".to_string(),
            }),
            ("Some", _) => Err(LustError::RuntimeError {
                message: "Lust Option.Some must carry exactly one payload".to_string(),
            }),
            (other, _) => Err(LustError::RuntimeError {
                message: format!("Unexpected Lust Option variant '{other}'"),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Option(inner) => T::matches_lust_type(inner),
            TypeKind::Unknown => true,
            TypeKind::Union(types) => types.iter().any(Self::matches_lust_type),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "Option"
    }
}

impl<T, E> IntoLustValue for core::result::Result<T, E>
where
    T: IntoLustValue,
    E: IntoLustValue,
{
    fn into_value(self) -> Value {
        match self {
            Ok(value) => Value::enum_variant("Result", "Ok", vec![value.into_value()]),
            Err(error) => Value::enum_variant("Result", "Err", vec![error.into_value()]),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Result(ok, err) => T::matches_lust_type(ok) && E::matches_lust_type(err),
            TypeKind::Unknown => true,
            TypeKind::Union(types) => types.iter().any(Self::matches_lust_type),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "Result"
    }
}

impl<T, E> FromLustValue for core::result::Result<T, E>
where
    T: FromLustValue,
    E: FromLustValue,
{
    fn from_value(value: Value) -> Result<Self> {
        let (variant, payload) = expected_enum_parts(&value, "Result")?;
        match (variant, payload) {
            ("Ok", Some([value])) => T::from_value(value.clone()).map(Ok),
            ("Err", Some([error])) => E::from_value(error.clone()).map(Err),
            ("Ok", _) => Err(LustError::RuntimeError {
                message: "Lust Result.Ok must carry exactly one payload".to_string(),
            }),
            ("Err", _) => Err(LustError::RuntimeError {
                message: "Lust Result.Err must carry exactly one payload".to_string(),
            }),
            (other, _) => Err(LustError::RuntimeError {
                message: format!("Unexpected Lust Result variant '{other}'"),
            }),
        }
    }

    fn matches_lust_type(ty: &Type) -> bool {
        match &ty.kind {
            TypeKind::Result(ok, err) => T::matches_lust_type(ok) && E::matches_lust_type(err),
            TypeKind::Unknown => true,
            TypeKind::Union(types) => types.iter().any(Self::matches_lust_type),
            _ => false,
        }
    }

    fn type_description() -> &'static str {
        "Result"
    }
}

fn expected_enum_parts<'a>(
    value: &'a Value,
    expected: &str,
) -> Result<(&'a str, Option<&'a [Value]>)> {
    let Some((enum_name, variant, payload)) = value.as_enum() else {
        return Err(LustError::RuntimeError {
            message: format!("Expected Lust enum '{expected}' but received '{:?}'", value),
        });
    };
    if enum_name != expected {
        return Err(LustError::RuntimeError {
            message: format!("Expected Lust enum '{expected}' but received '{enum_name}'"),
        });
    }
    Ok((variant, payload))
}

impl<T> FunctionArgs for T
where
    T: IntoLustValue,
{
    fn into_values(self) -> Vec<Value> {
        if T::is_unit() {
            Vec::new()
        } else {
            vec![self.into_value()]
        }
    }

    fn validate_signature(function_name: &str, params: &[Type]) -> Result<()> {
        if T::is_unit() {
            ensure_arity(function_name, params, 0)
        } else {
            ensure_arity(function_name, params, 1)?;
            ensure_arg_type::<T>(function_name, params, 0)
        }
    }
}

macro_rules! impl_function_args {
    ($($index:tt: $T:ident),+ $(,)?) => {
        impl<$($T),+> FunctionArgs for ($($T,)+)
        where
            $($T: IntoLustValue,)+
        {
            fn into_values(self) -> Vec<Value> {
                vec![$(self.$index.into_value()),+]
            }

            fn validate_signature(function_name: &str, params: &[Type]) -> Result<()> {
                const ARITY: usize = count_idents!($($T),+);
                ensure_arity(function_name, params, ARITY)?;
                $(ensure_arg_type::<$T>(function_name, params, $index)?;)+
                Ok(())
            }
        }
    };
}

impl_function_args!(0: A, 1: B);
impl_function_args!(0: A, 1: B, 2: C);
impl_function_args!(0: A, 1: B, 2: C, 3: D);
impl_function_args!(0: A, 1: B, 2: C, 3: D, 4: E);
impl_function_args!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F);
impl_function_args!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G);
impl_function_args!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H);
impl_function_args!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I);
impl_function_args!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J);
impl_function_args!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K);
impl_function_args!(0: A, 1: B, 2: C, 3: D, 4: E, 5: F, 6: G, 7: H, 8: I, 9: J, 10: K, 11: L);

fn ensure_arity(function_name: &str, params: &[Type], provided: usize) -> Result<()> {
    if params.len() == provided {
        Ok(())
    } else {
        Err(LustError::TypeError {
            message: format!(
                "Function '{}' expects {} argument(s) but {} were supplied",
                function_name,
                params.len(),
                provided
            ),
        })
    }
}

fn ensure_arg_type<T: IntoLustValue>(
    function_name: &str,
    params: &[Type],
    index: usize,
) -> Result<()> {
    if <T as IntoLustValue>::matches_lust_type(&params[index]) {
        Ok(())
    } else {
        Err(argument_type_mismatch(
            function_name,
            index,
            <T as IntoLustValue>::type_description(),
            &params[index],
        ))
    }
}

fn argument_type_mismatch(
    function_name: &str,
    index: usize,
    rust_type: &str,
    lust_type: &Type,
) -> LustError {
    LustError::TypeError {
        message: format!(
            "Function '{}' parameter {} expects Lust type '{}' but Rust provided '{}'",
            function_name,
            index + 1,
            lust_type,
            rust_type
        ),
    }
}

#[cfg(test)]
mod function_args_tests {
    use crate::{EmbeddedProgram, LustInt};

    #[test]
    fn typed_calls_preserve_order_for_six_and_twelve_arguments() {
        let mut program = EmbeddedProgram::builder()
            .module(
                "main",
                r#"
function six(a: int, b: int, c: int, d: int, e: int, f: int): int
    return a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f
end

function twelve(a: int, b: int, c: int, d: int, e: int, f: int,
                g: int, h: int, i: int, j: int, k: int, l: int): int
    return a * 100000000000 + b * 10000000000 + c * 1000000000 + d * 100000000 +
           e * 10000000 + f * 1000000 + g * 100000 + h * 10000 + i * 1000 +
           j * 100 + k * 10 + l
end
"#,
            )
            .entry_module("main")
            .compile()
            .expect("compile typed-call fixture");

        let six: LustInt = program
            .call_typed("main.six", (1, 2, 3, 4, 5, 6))
            .expect("call with six arguments");
        assert_eq!(six, 123456);

        let twelve_handle = program
            .function_handle("main.twelve")
            .expect("get twelve-argument handle");
        let twelve: LustInt = twelve_handle
            .call_typed(&mut program, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
            .expect("call handle with twelve arguments");
        assert_eq!(twelve, 123456790122);

        let arity_error = program
            .call_typed::<_, LustInt>("main.six", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
            .expect_err("wrong tuple arity should be rejected before indexed checks");
        assert!(
            arity_error
                .to_string()
                .contains("expects 6 argument(s) but 12 were supplied")
        );

        let type_error = program
            .call_typed::<_, LustInt>(
                "main.twelve",
                (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, String::from("wrong")),
            )
            .expect_err("the final tuple slot should be validated");
        assert!(type_error.to_string().contains("parameter 12"));
    }
}
