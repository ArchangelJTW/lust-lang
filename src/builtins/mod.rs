use crate::FunctionSignature;
use crate::ast::{Span, Type, TypeKind};
use crate::lazy::StaticOnceCell;
use alloc::{boxed::Box, collections::BTreeMap, string::ToString, vec, vec::Vec};
use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct BuiltinSignature {
    pub type_params: &'static [&'static str],
    pub params: Vec<TypeExpr>,
    pub return_type: TypeExpr,
}

#[derive(Debug, Clone)]
pub struct BuiltinFunction {
    pub name: &'static str,
    pub description: &'static str,
    pub signature: BuiltinSignature,
    pub param_names: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MethodSemantics {
    Simple,
    ArrayMap,
    ArrayFilter,
    ArrayReduce,
}

#[derive(Debug, Clone)]
pub struct BuiltinMethod {
    pub receiver: TypeExpr,
    pub name: &'static str,
    pub description: &'static str,
    pub signature: BuiltinSignature,
    pub param_names: &'static [&'static str],
    pub semantics: MethodSemantics,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Int,
    Float,
    Bool,
    String,
    Unit,
    Unknown,
    Named(&'static str),
    Array(Box<TypeExpr>),
    Map(Box<TypeExpr>, Box<TypeExpr>),
    Result(Box<TypeExpr>, Box<TypeExpr>),
    Option(Box<TypeExpr>),
    Generic(&'static str),
    SelfType,
    Function {
        params: Vec<TypeExpr>,
        return_type: Box<TypeExpr>,
    },
}

impl BuiltinFunction {
    pub fn to_signature(&self, span: Span) -> FunctionSignature {
        FunctionSignature {
            params: self
                .signature
                .params
                .iter()
                .map(|expr| expr.instantiate(&HashMap::new(), Some(span)))
                .collect(),
            return_type: self
                .signature
                .return_type
                .instantiate(&HashMap::new(), Some(span)),
            is_method: false,
            type_params: self
                .signature
                .type_params
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            trait_bounds: Vec::new(),
            effects: None,
        }
    }

    pub fn parameters(&self) -> Vec<(&'static str, &TypeExpr)> {
        self.signature
            .params
            .iter()
            .enumerate()
            .map(|(idx, ty)| {
                let name = self.param_names.get(idx).copied().unwrap_or("");
                (name, ty)
            })
            .collect()
    }

    pub fn return_type(&self) -> &TypeExpr {
        &self.signature.return_type
    }
}

impl BuiltinMethod {
    pub fn parameters(&self) -> Vec<(&'static str, &TypeExpr)> {
        self.signature
            .params
            .iter()
            .enumerate()
            .map(|(idx, ty)| {
                let name = self.param_names.get(idx).copied().unwrap_or("");
                (name, ty)
            })
            .collect()
    }

    pub fn return_type(&self) -> &TypeExpr {
        &self.signature.return_type
    }

    pub fn receiver_type(&self) -> &TypeExpr {
        &self.receiver
    }
}

impl TypeExpr {
    pub fn instantiate(&self, generics: &HashMap<&'static str, Type>, span: Option<Span>) -> Type {
        let span = span.unwrap_or_else(Span::dummy);
        match self {
            TypeExpr::Int => Type::new(TypeKind::Int, span),
            TypeExpr::Float => Type::new(TypeKind::Float, span),
            TypeExpr::Bool => Type::new(TypeKind::Bool, span),
            TypeExpr::String => Type::new(TypeKind::String, span),
            TypeExpr::Unit => Type::new(TypeKind::Unit, span),
            TypeExpr::Unknown => Type::new(TypeKind::Unknown, span),
            TypeExpr::Named(name) => Type::new(TypeKind::Named((*name).to_string()), span),
            TypeExpr::Array(inner) => Type::new(
                TypeKind::Array(Box::new(inner.instantiate(generics, Some(span)))),
                span,
            ),
            TypeExpr::Map(key, value) => Type::new(
                TypeKind::Map(
                    Box::new(key.instantiate(generics, Some(span))),
                    Box::new(value.instantiate(generics, Some(span))),
                ),
                span,
            ),
            TypeExpr::Result(ok, err) => Type::new(
                TypeKind::Result(
                    Box::new(ok.instantiate(generics, Some(span))),
                    Box::new(err.instantiate(generics, Some(span))),
                ),
                span,
            ),
            TypeExpr::Option(inner) => Type::new(
                TypeKind::Option(Box::new(inner.instantiate(generics, Some(span)))),
                span,
            ),
            TypeExpr::Generic(name) => generics
                .get(name)
                .cloned()
                .unwrap_or_else(|| Type::new(TypeKind::Generic((*name).to_string()), span)),
            TypeExpr::SelfType => generics
                .get("Self")
                .cloned()
                .unwrap_or_else(|| Type::new(TypeKind::Unknown, span)),
            TypeExpr::Function {
                params,
                return_type,
            } => Type::new(
                TypeKind::Function {
                    params: params
                        .iter()
                        .map(|param| param.instantiate(generics, Some(span)))
                        .collect(),
                    return_type: Box::new(return_type.instantiate(generics, Some(span))),
                },
                span,
            ),
        }
    }
}

fn match_type_expr(
    pattern: &TypeExpr,
    actual: &Type,
    bindings: &mut HashMap<&'static str, Type>,
) -> bool {
    match (pattern, &actual.kind) {
        (TypeExpr::SelfType, _) => {
            bindings.insert("Self", actual.clone());
            true
        }
        (TypeExpr::Generic(name), _) => {
            if let Some(existing) = bindings.get(name) {
                existing.kind == actual.kind
            } else {
                bindings.insert(name, actual.clone());
                true
            }
        }
        (TypeExpr::Int, TypeKind::Int) => true,
        (TypeExpr::Float, TypeKind::Float) => true,
        (TypeExpr::Bool, TypeKind::Bool) => true,
        (TypeExpr::String, TypeKind::String) => true,
        (TypeExpr::Unit, TypeKind::Unit) => true,
        (TypeExpr::Unknown, TypeKind::Unknown) => true,
        (TypeExpr::Named(expected), TypeKind::Named(actual_name)) => expected == actual_name,
        (TypeExpr::Array(pattern_inner), TypeKind::Array(actual_inner)) => {
            match_type_expr(pattern_inner, actual_inner, bindings)
        }
        (TypeExpr::Map(pattern_key, pattern_value), TypeKind::Map(actual_key, actual_value)) => {
            match_type_expr(pattern_key, actual_key, bindings)
                && match_type_expr(pattern_value, actual_value, bindings)
        }
        (TypeExpr::Option(pattern_inner), TypeKind::Option(actual_inner)) => {
            match_type_expr(pattern_inner, actual_inner, bindings)
        }
        (TypeExpr::Result(pattern_ok, pattern_err), TypeKind::Result(actual_ok, actual_err)) => {
            match_type_expr(pattern_ok, actual_ok, bindings)
                && match_type_expr(pattern_err, actual_err, bindings)
        }
        _ => false,
    }
}

pub fn match_receiver(pattern: &TypeExpr, actual: &Type) -> Option<HashMap<&'static str, Type>> {
    let mut bindings = HashMap::new();
    if match_type_expr(pattern, actual, &mut bindings) {
        Some(bindings)
    } else {
        None
    }
}

fn func(
    name: &'static str,
    description: &'static str,
    param_names: &'static [&'static str],
    params: Vec<TypeExpr>,
    return_type: TypeExpr,
) -> BuiltinFunction {
    BuiltinFunction {
        name,
        description,
        signature: BuiltinSignature {
            type_params: &[],
            params,
            return_type,
        },
        param_names,
    }
}

fn generic_func(
    name: &'static str,
    description: &'static str,
    type_params: &'static [&'static str],
    param_names: &'static [&'static str],
    params: Vec<TypeExpr>,
    return_type: TypeExpr,
) -> BuiltinFunction {
    BuiltinFunction {
        name,
        description,
        signature: BuiltinSignature {
            type_params,
            params,
            return_type,
        },
        param_names,
    }
}

fn method(
    receiver: TypeExpr,
    name: &'static str,
    description: &'static str,
    param_names: &'static [&'static str],
    params: Vec<TypeExpr>,
    return_type: TypeExpr,
) -> BuiltinMethod {
    BuiltinMethod {
        receiver,
        name,
        description,
        signature: BuiltinSignature {
            type_params: &[],
            params,
            return_type,
        },
        param_names,
        semantics: MethodSemantics::Simple,
    }
}

fn iterator_methods() -> Vec<BuiltinMethod> {
    vec![
        method(
            TypeExpr::Named("Iterator"),
            "iter",
            "Return the iterator itself",
            &[],
            vec![],
            TypeExpr::Named("Iterator"),
        ),
        method(
            TypeExpr::Named("Iterator"),
            "next",
            "Advance the iterator and return the next value",
            &[],
            vec![],
            TypeExpr::Option(Box::new(TypeExpr::Unknown)),
        ),
    ]
}

fn option_methods() -> Vec<BuiltinMethod> {
    let receiver = TypeExpr::Option(Box::new(TypeExpr::Generic("T")));
    vec![
        method(
            receiver.clone(),
            "is_some",
            "Check if the option contains a value",
            &[],
            vec![],
            TypeExpr::Bool,
        ),
        method(
            receiver.clone(),
            "is_none",
            "Check if the option is empty",
            &[],
            vec![],
            TypeExpr::Bool,
        ),
        method(
            receiver.clone(),
            "unwrap",
            "Unwrap the contained value, panicking if None",
            &[],
            vec![],
            TypeExpr::Generic("T"),
        ),
        method(
            receiver,
            "unwrap_or",
            "Return the value or a provided default",
            &["default"],
            vec![TypeExpr::Generic("T")],
            TypeExpr::Generic("T"),
        ),
    ]
}

fn result_methods() -> Vec<BuiltinMethod> {
    let receiver = TypeExpr::Result(
        Box::new(TypeExpr::Generic("T")),
        Box::new(TypeExpr::Generic("E")),
    );
    vec![
        method(
            receiver.clone(),
            "is_ok",
            "Check if the result is Ok",
            &[],
            vec![],
            TypeExpr::Bool,
        ),
        method(
            receiver.clone(),
            "is_err",
            "Check if the result is Err",
            &[],
            vec![],
            TypeExpr::Bool,
        ),
        method(
            receiver.clone(),
            "unwrap",
            "Unwrap the Ok value, panicking if Err",
            &[],
            vec![],
            TypeExpr::Generic("T"),
        ),
        method(
            receiver,
            "unwrap_or",
            "Return the Ok value or a provided default",
            &["default"],
            vec![TypeExpr::Generic("T")],
            TypeExpr::Generic("T"),
        ),
    ]
}

static BASE_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();

fn build_base_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "print",
            "Print values without a newline",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unit,
        ),
        func(
            "println",
            "Print values followed by a newline",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unit,
        ),
        func(
            "type",
            "Return the runtime type name",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::String,
        ),
        func(
            "tostring",
            "Convert a value to a string",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::String,
        ),
        func(
            "error",
            "Raise a runtime error",
            &["message"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "assert",
            "Assert a condition or raise an error",
            &["cond", "message"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "tonumber",
            "Convert a value to a number",
            &["value", "base"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "pairs",
            "Iterate over key/value pairs",
            &["table"],
            vec![TypeExpr::Unknown],
            TypeExpr::Named("Iterator"),
        ),
        func(
            "ipairs",
            "Iterate over array elements with indices",
            &["array"],
            vec![TypeExpr::Unknown],
            TypeExpr::Named("Iterator"),
        ),
        func(
            "select",
            "Return arguments starting at an index or the argument count",
            &["index_or_hash", "..."],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "random",
            "Generate a random number in an optional range",
            &["m", "n"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "randomseed",
            "Seed the random number generator",
            &["seed"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unit,
        ),
        func(
            "unpack",
            "Unpack array elements into multiple returns",
            &["table", "i", "j"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "setmetatable",
            "Assign a metatable to a Lua table value",
            &["table", "meta"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
    ]
}

static ARRAY_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static TABLE_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static MAP_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static MATH_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static STRING_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static TASK_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static LUA_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static IO_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static OS_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static RX_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static TEST_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static RANDOM_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();

fn build_array_functions() -> Vec<BuiltinFunction> {
    let t = TypeExpr::Generic("T");
    let arr_t = TypeExpr::Array(Box::new(t.clone()));
    vec![
        generic_func(
            "array.len",
            "Return the number of elements in the array",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Int,
        ),
        generic_func(
            "array.is_empty",
            "Check if the array contains no elements",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Bool,
        ),
        generic_func(
            "array.get",
            "Return the element at the given index, if any",
            &["T"],
            &["arr", "index"],
            vec![arr_t.clone(), TypeExpr::Int],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "array.first",
            "Return the first element, if any",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "array.last",
            "Return the last element, if any",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "array.push",
            "Append a value to the array",
            &["T"],
            &["arr", "value"],
            vec![arr_t.clone(), t.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "array.pop",
            "Remove and return the last element, if any",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "array.insert",
            "Insert an element at the given index",
            &["T"],
            &["arr", "index", "value"],
            vec![arr_t.clone(), TypeExpr::Int, t.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "array.remove",
            "Remove and return the element at the given index",
            &["T"],
            &["arr", "index"],
            vec![arr_t.clone(), TypeExpr::Int],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "array.clear",
            "Remove all elements from the array",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "array.slice",
            "Return a slice of the array between two indices",
            &["T"],
            &["arr", "start", "end"],
            vec![arr_t.clone(), TypeExpr::Int, TypeExpr::Int],
            arr_t.clone(),
        ),
        generic_func(
            "array.concat",
            "Concatenate array elements into a string with an optional separator",
            &["T"],
            &["arr", "sep"],
            vec![arr_t.clone(), TypeExpr::Unknown],
            TypeExpr::String,
        ),
        generic_func(
            "array.sort",
            "Sort elements in the array in-place",
            &["T"],
            &["arr", "comp"],
            vec![arr_t.clone(), TypeExpr::Unknown],
            TypeExpr::Unit,
        ),
        generic_func(
            "array.reverse",
            "Reverse elements in the array in-place",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "array.contains",
            "Check if the array contains a given value",
            &["T"],
            &["arr", "value"],
            vec![arr_t.clone(), t.clone()],
            TypeExpr::Bool,
        ),
        generic_func(
            "array.map",
            "Transform each element using the provided function",
            &["T"],
            &["arr", "func"],
            vec![
                arr_t.clone(),
                TypeExpr::Function {
                    params: vec![t.clone()],
                    return_type: Box::new(TypeExpr::Unknown),
                },
            ],
            TypeExpr::Array(Box::new(TypeExpr::Unknown)),
        ),
        generic_func(
            "array.filter",
            "Keep elements where the predicate returns true",
            &["T"],
            &["arr", "func"],
            vec![
                arr_t.clone(),
                TypeExpr::Function {
                    params: vec![t.clone()],
                    return_type: Box::new(TypeExpr::Bool),
                },
            ],
            arr_t.clone(),
        ),
        generic_func(
            "array.reduce",
            "Fold elements into a single value",
            &["T"],
            &["arr", "initial", "func"],
            vec![
                arr_t.clone(),
                TypeExpr::Unknown,
                TypeExpr::Function {
                    params: vec![TypeExpr::Unknown, t.clone()],
                    return_type: Box::new(TypeExpr::Unknown),
                },
            ],
            TypeExpr::Unknown,
        ),
        generic_func(
            "array.iter",
            "Return an iterator over the array items",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Named("Iterator"),
        ),
    ]
}

fn build_table_functions() -> Vec<BuiltinFunction> {
    let t = TypeExpr::Generic("T");
    let arr_t = TypeExpr::Array(Box::new(t.clone()));
    vec![
        generic_func(
            "table.len",
            "Return the number of elements in the array/table",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Int,
        ),
        generic_func(
            "table.is_empty",
            "Check if the array/table contains no elements",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Bool,
        ),
        generic_func(
            "table.get",
            "Return the element at the given index, if any",
            &["T"],
            &["arr", "index"],
            vec![arr_t.clone(), TypeExpr::Int],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "table.first",
            "Return the first element, if any",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "table.last",
            "Return the last element, if any",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "table.push",
            "Append a value to the array/table",
            &["T"],
            &["arr", "value"],
            vec![arr_t.clone(), t.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "table.pop",
            "Remove and return the last element, if any",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "table.insert",
            "Insert a value at index or push if 2 arguments are provided",
            &["T"],
            &["arr", "index_or_val", "value"],
            vec![arr_t.clone(), TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unit,
        ),
        generic_func(
            "table.remove",
            "Remove element at index, or remove last element if 1 argument",
            &["T"],
            &["arr", "index"],
            vec![arr_t.clone(), TypeExpr::Unknown],
            TypeExpr::Option(Box::new(t.clone())),
        ),
        generic_func(
            "table.concat",
            "Concatenate array/table elements into a string with an optional separator",
            &["T"],
            &["arr", "sep"],
            vec![arr_t.clone(), TypeExpr::Unknown],
            TypeExpr::String,
        ),
        generic_func(
            "table.sort",
            "Sort elements in the array/table in-place",
            &["T"],
            &["arr", "comp"],
            vec![arr_t.clone(), TypeExpr::Unknown],
            TypeExpr::Unit,
        ),
        generic_func(
            "table.reverse",
            "Reverse elements in the array/table in-place",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "table.clear",
            "Remove all elements from the array/table",
            &["T"],
            &["arr"],
            vec![arr_t.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "table.contains",
            "Check if the array/table contains a given value",
            &["T"],
            &["arr", "value"],
            vec![arr_t.clone(), t.clone()],
            TypeExpr::Bool,
        ),
        generic_func(
            "table.slice",
            "Return a slice of the array/table between two indices",
            &["T"],
            &["arr", "start", "end"],
            vec![arr_t.clone(), TypeExpr::Int, TypeExpr::Int],
            arr_t.clone(),
        ),
    ]
}

fn build_map_functions() -> Vec<BuiltinFunction> {
    let k = TypeExpr::Generic("K");
    let v = TypeExpr::Generic("V");
    let map_kv = TypeExpr::Map(Box::new(k.clone()), Box::new(v.clone()));
    vec![
        generic_func(
            "map.len",
            "Return the number of entries in the map",
            &["K", "V"],
            &["map"],
            vec![map_kv.clone()],
            TypeExpr::Int,
        ),
        generic_func(
            "map.is_empty",
            "Check if the map contains no entries",
            &["K", "V"],
            &["map"],
            vec![map_kv.clone()],
            TypeExpr::Bool,
        ),
        generic_func(
            "map.get",
            "Look up a value by key",
            &["K", "V"],
            &["map", "key"],
            vec![map_kv.clone(), k.clone()],
            TypeExpr::Option(Box::new(v.clone())),
        ),
        generic_func(
            "map.set",
            "Insert or overwrite a key/value pair",
            &["K", "V"],
            &["map", "key", "value"],
            vec![map_kv.clone(), k.clone(), v.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "map.has",
            "Check whether the map contains a key",
            &["K", "V"],
            &["map", "key"],
            vec![map_kv.clone(), k.clone()],
            TypeExpr::Bool,
        ),
        generic_func(
            "map.delete",
            "Remove an entry from the map and return its value if present",
            &["K", "V"],
            &["map", "key"],
            vec![map_kv.clone(), k.clone()],
            TypeExpr::Option(Box::new(v.clone())),
        ),
        generic_func(
            "map.clear",
            "Remove all entries from the map",
            &["K", "V"],
            &["map"],
            vec![map_kv.clone()],
            TypeExpr::Unit,
        ),
        generic_func(
            "map.keys",
            "Return the keys of the map as an array",
            &["K", "V"],
            &["map"],
            vec![map_kv.clone()],
            TypeExpr::Array(Box::new(k.clone())),
        ),
        generic_func(
            "map.values",
            "Return the values of the map as an array",
            &["K", "V"],
            &["map"],
            vec![map_kv.clone()],
            TypeExpr::Array(Box::new(v.clone())),
        ),
        generic_func(
            "map.iter",
            "Iterate over key/value pairs in the map",
            &["K", "V"],
            &["map"],
            vec![map_kv.clone()],
            TypeExpr::Named("Iterator"),
        ),
    ]
}

fn build_math_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "math.abs",
            "Return the absolute value of a number",
            &["x"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "math.floor",
            "Return the largest integer less than or equal to x",
            &["x"],
            vec![TypeExpr::Unknown],
            TypeExpr::Int,
        ),
        func(
            "math.ceil",
            "Return the smallest integer greater than or equal to x",
            &["x"],
            vec![TypeExpr::Unknown],
            TypeExpr::Int,
        ),
        func(
            "math.round",
            "Round to the nearest integer",
            &["x"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "math.sqrt",
            "Return the square root of x",
            &["x"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.sin",
            "Return the sine of x in radians",
            &["x"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.cos",
            "Return the cosine of x in radians",
            &["x"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.tan",
            "Return the tangent of x in radians",
            &["x"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.asin",
            "Return the arcsine of x in radians",
            &["x"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.acos",
            "Return the arccosine of x in radians",
            &["x"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.atan",
            "Return the arctangent of x in radians",
            &["x"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.atan2",
            "Return the arctangent of y/x in radians",
            &["y", "x"],
            vec![TypeExpr::Float, TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.min",
            "Return the minimum of the provided numbers",
            &["x", "y"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "math.max",
            "Return the maximum of the provided numbers",
            &["x", "y"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "math.clamp",
            "Clamp a number between min and max bounds",
            &["x", "min", "max"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "math.random",
            "Generate a pseudo-random number",
            &["m", "n"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "math.randomseed",
            "Set the seed for the pseudo-random number generator",
            &["seed"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unit,
        ),
        func(
            "math.deg",
            "Convert angle from radians to degrees",
            &["rad"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.rad",
            "Convert angle from degrees to radians",
            &["deg"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.exp",
            "Return e^x",
            &["x"],
            vec![TypeExpr::Float],
            TypeExpr::Float,
        ),
        func(
            "math.log",
            "Return the natural logarithm of x (or logarithm to optional base)",
            &["x", "base"],
            vec![TypeExpr::Float, TypeExpr::Unknown],
            TypeExpr::Float,
        ),
        func(
            "math.tointeger",
            "Convert a number to an integer if possible",
            &["x"],
            vec![TypeExpr::Unknown],
            TypeExpr::Option(Box::new(TypeExpr::Int)),
        ),
        func(
            "math.tofloat",
            "Convert a number to a float",
            &["x"],
            vec![TypeExpr::Unknown],
            TypeExpr::Option(Box::new(TypeExpr::Float)),
        ),
        func(
            "math.mod",
            "Floored modulo with divisor sign",
            &["a", "b"],
            vec![TypeExpr::Int, TypeExpr::Int],
            TypeExpr::Int,
        ),
    ]
}

fn build_string_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "string.len",
            "Return the length of a string in bytes",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::Int,
        ),
        func(
            "string.sub",
            "Extract a substring using 1-based indices",
            &["s", "i", "j"],
            vec![TypeExpr::String, TypeExpr::Int, TypeExpr::Unknown],
            TypeExpr::String,
        ),
        func(
            "string.lower",
            "Convert a string to lowercase",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::String,
        ),
        func(
            "string.upper",
            "Convert a string to uppercase",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::String,
        ),
        func(
            "string.byte",
            "Return internal numeric codes of characters",
            &["s", "i", "j"],
            vec![TypeExpr::String, TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "string.char",
            "Create a string from integer character codes",
            &["byte"],
            vec![TypeExpr::Unknown],
            TypeExpr::String,
        ),
        func(
            "string.find",
            "Find the first match of pattern in string",
            &["s", "pattern", "init", "plain"],
            vec![
                TypeExpr::String,
                TypeExpr::String,
                TypeExpr::Unknown,
                TypeExpr::Unknown,
            ],
            TypeExpr::Unknown,
        ),
        func(
            "string.match",
            "Extract pattern captures from string",
            &["s", "pattern", "init"],
            vec![TypeExpr::String, TypeExpr::String, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "string.gsub",
            "Global pattern substitution in string",
            &["s", "pattern", "repl", "n"],
            vec![
                TypeExpr::String,
                TypeExpr::String,
                TypeExpr::Unknown,
                TypeExpr::Unknown,
            ],
            TypeExpr::Unknown,
        ),
        func(
            "string.format",
            "Format a string using printf-style specifiers",
            &["fmt", "..."],
            vec![TypeExpr::String, TypeExpr::Unknown],
            TypeExpr::String,
        ),
        func(
            "string.rep",
            "Return a string repeated n times",
            &["s", "n", "sep"],
            vec![TypeExpr::String, TypeExpr::Int, TypeExpr::Unknown],
            TypeExpr::String,
        ),
        func(
            "string.reverse",
            "Return a string with reversed characters",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::String,
        ),
        func(
            "string.split",
            "Split a string by delimiter",
            &["s", "delimiter"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::Array(Box::new(TypeExpr::String)),
        ),
        func(
            "string.trim",
            "Trim whitespace from start and end of string",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::String,
        ),
        func(
            "string.trim_start",
            "Trim leading whitespace from string",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::String,
        ),
        func(
            "string.trim_end",
            "Trim trailing whitespace from string",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::String,
        ),
        func(
            "string.replace",
            "Replace occurrences of substring with replacement",
            &["s", "from", "to"],
            vec![TypeExpr::String, TypeExpr::String, TypeExpr::String],
            TypeExpr::String,
        ),
        func(
            "string.starts_with",
            "Check if string starts with prefix",
            &["s", "prefix"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::Bool,
        ),
        func(
            "string.ends_with",
            "Check if string ends with suffix",
            &["s", "suffix"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::Bool,
        ),
        func(
            "string.contains",
            "Check if string contains substring",
            &["s", "substring"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::Bool,
        ),
        func(
            "string.is_empty",
            "Check if string is empty",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::Bool,
        ),
        func(
            "string.chars",
            "Return characters of string as array of single-character strings",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::Array(Box::new(TypeExpr::String)),
        ),
        func(
            "string.lines",
            "Return lines of string as array of strings",
            &["s"],
            vec![TypeExpr::String],
            TypeExpr::Array(Box::new(TypeExpr::String)),
        ),
        func(
            "string.slice",
            "Return a 0-based slice of string [start..end)",
            &["s", "start", "end"],
            vec![TypeExpr::String, TypeExpr::Int, TypeExpr::Int],
            TypeExpr::String,
        ),
        func(
            "string.index_of",
            "Return 0-based index of substring, or None",
            &["s", "substr"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::Option(Box::new(TypeExpr::Int)),
        ),
    ]
}
fn build_task_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "task.run",
            "Run a function as a task",
            &["func"],
            vec![TypeExpr::Unknown],
            TypeExpr::Named("Task"),
        ),
        func(
            "task.create",
            "Create a suspended task",
            &["func"],
            vec![TypeExpr::Unknown],
            TypeExpr::Named("Task"),
        ),
        func(
            "task.status",
            "Get the status of a task",
            &["task"],
            vec![TypeExpr::Named("Task")],
            TypeExpr::Named("TaskStatus"),
        ),
        func(
            "task.info",
            "Get detailed information about a task",
            &["task"],
            vec![TypeExpr::Named("Task")],
            TypeExpr::Named("TaskInfo"),
        ),
        func(
            "task.resume",
            "Resume a suspended task",
            &["task"],
            vec![TypeExpr::Named("Task")],
            TypeExpr::Named("TaskInfo"),
        ),
        func(
            "task.yield",
            "Yield from the current task",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "task.stop",
            "Stop a running task",
            &["task"],
            vec![TypeExpr::Named("Task")],
            TypeExpr::Bool,
        ),
        func(
            "task.restart",
            "Restart a completed task",
            &["task"],
            vec![TypeExpr::Named("Task")],
            TypeExpr::Named("TaskInfo"),
        ),
        func(
            "task.current",
            "Return the currently executing task",
            &[],
            vec![],
            TypeExpr::Option(Box::new(TypeExpr::Named("Task"))),
        ),
    ]
}

fn build_lua_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "lua.to_value",
            "Wrap a Lust value in LuaValue",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::Named("LuaValue"),
        ),
        func(
            "lua.require",
            "Lua-style module resolver (loads from already-initialized globals when available)",
            &["name"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "lua.table",
            "Create an empty Lua-style table",
            &[],
            vec![],
            TypeExpr::Named("LuaTable"),
        ),
        func(
            "lua.setmetatable",
            "Set the metatable for a Lua table value",
            &["table", "meta"],
            vec![TypeExpr::Named("LuaValue"), TypeExpr::Named("LuaValue")],
            TypeExpr::Named("LuaValue"),
        ),
        func(
            "lua.getmetatable",
            "Get the metatable for a Lua table value",
            &["table"],
            vec![TypeExpr::Named("LuaValue")],
            TypeExpr::Named("LuaValue"),
        ),
        func(
            "lua.unwrap",
            "Extract a raw Lust value from a LuaValue wrapper",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
    ]
}

fn build_io_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "io.read_file",
            "Read the contents of a file",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Result(Box::new(TypeExpr::String), Box::new(TypeExpr::String)),
        ),
        func(
            "io.read_dir",
            "Read the file names of a directory",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Result(
                Box::new(TypeExpr::Array(Box::new(TypeExpr::String))),
                Box::new(TypeExpr::String),
            ),
        ),
        func(
            "io.read_file_bytes",
            "Read the contents of a file as byte values",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Result(
                Box::new(TypeExpr::Array(Box::new(TypeExpr::Int))),
                Box::new(TypeExpr::String),
            ),
        ),
        func(
            "io.write_file",
            "Write contents to a file",
            &["path", "value"],
            vec![TypeExpr::String, TypeExpr::Unknown],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
        ),
        func(
            "io.read_stdin",
            "Read all available stdin",
            &[],
            vec![],
            TypeExpr::Result(Box::new(TypeExpr::String), Box::new(TypeExpr::String)),
        ),
        func(
            "io.read_line",
            "Read a single line from stdin",
            &[],
            vec![],
            TypeExpr::Result(Box::new(TypeExpr::String), Box::new(TypeExpr::String)),
        ),
        func(
            "io.write_stdout",
            "Write a value to stdout",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
        ),
        func(
            "io.read_file_effect",
            "Read file contents returning an Effect",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "io.write_file_effect",
            "Write file contents returning an Effect",
            &["path", "content"],
            vec![TypeExpr::String, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "io.exists_effect",
            "Check path existence returning an Effect",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "io.remove_file_effect",
            "Remove a file returning an Effect",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "io.create_dir_all_effect",
            "Create directories recursively returning an Effect",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "io.read_dir_effect",
            "Read directory entries returning an Effect",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "io.open_scoped",
            "Open scoped file handle returning an Effect",
            &["path", "mode"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "io.read_handle",
            "Read bytes from handle returning an Effect",
            &["handle", "max_bytes"],
            vec![TypeExpr::Unknown, TypeExpr::Int],
            TypeExpr::Unknown,
        ),
        func(
            "io.write_handle",
            "Write bytes to handle returning an Effect",
            &["handle", "bytes"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "io.close_handle",
            "Close file handle returning an Effect",
            &["handle"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
    ]
}

fn build_os_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "os.time",
            "Get the current UNIX timestamp with sub-second precision",
            &[],
            vec![],
            TypeExpr::Float,
        ),
        func(
            "os.date",
            "Format date and time or return date components table",
            &["format", "time"],
            vec![TypeExpr::Option(Box::new(TypeExpr::String)), TypeExpr::Option(Box::new(TypeExpr::Float))],
            TypeExpr::Unknown,
        ),
        func(
            "os.sleep",
            "Sleep for the given number of seconds",
            &["seconds"],
            vec![TypeExpr::Float],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
        ),
        func(
            "os.create_file",
            "Create an empty file on disk",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
        ),
        func(
            "os.create_dir",
            "Create a directory",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
        ),
        func(
            "os.remove_file",
            "Remove a file from disk",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
        ),
        func(
            "os.remove_dir",
            "Remove an empty directory",
            &["path"],
            vec![TypeExpr::String],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
        ),
        func(
            "os.rename",
            "Rename or move a path",
            &["from", "to"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
        ),
        func(
            "os.get_env",
            "Get an environment variable",
            &["name"],
            vec![TypeExpr::String],
            TypeExpr::Result(
                Box::new(TypeExpr::Option(Box::new(TypeExpr::String))),
                Box::new(TypeExpr::String),
            ),
        ),
        func(
            "os.getenv_effect",
            "Get an environment variable returning an Effect",
            &["key"],
            vec![TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "os.setenv_effect",
            "Set an environment variable returning an Effect",
            &["key", "val"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "os.now_ms_effect",
            "Get current time in milliseconds returning an Effect",
            &[],
            vec![],
            TypeExpr::Unknown,
        ),
        func(
            "os.args_effect",
            "Get command line arguments returning an Effect",
            &[],
            vec![],
            TypeExpr::Unknown,
        ),
    ]
}

static BUILTIN_METHODS: StaticOnceCell<Vec<BuiltinMethod>> = StaticOnceCell::new();

fn build_builtin_methods() -> Vec<BuiltinMethod> {
    let mut methods = Vec::new();
    methods.extend(iterator_methods());
    methods.extend(option_methods());
    methods.extend(result_methods());
    methods
}

pub fn base_functions() -> &'static [BuiltinFunction] {
    BASE_FUNCTIONS.get_or_init(build_base_functions).as_slice()
}

pub fn array_functions() -> &'static [BuiltinFunction] {
    ARRAY_FUNCTIONS
        .get_or_init(build_array_functions)
        .as_slice()
}

pub fn table_functions() -> &'static [BuiltinFunction] {
    TABLE_FUNCTIONS
        .get_or_init(build_table_functions)
        .as_slice()
}

pub fn map_functions() -> &'static [BuiltinFunction] {
    MAP_FUNCTIONS.get_or_init(build_map_functions).as_slice()
}

pub fn math_functions() -> &'static [BuiltinFunction] {
    MATH_FUNCTIONS.get_or_init(build_math_functions).as_slice()
}

pub fn string_functions() -> &'static [BuiltinFunction] {
    STRING_FUNCTIONS
        .get_or_init(build_string_functions)
        .as_slice()
}

pub fn task_functions() -> &'static [BuiltinFunction] {
    TASK_FUNCTIONS.get_or_init(build_task_functions).as_slice()
}

pub fn lua_functions() -> &'static [BuiltinFunction] {
    LUA_FUNCTIONS.get_or_init(build_lua_functions).as_slice()
}

pub fn io_functions() -> &'static [BuiltinFunction] {
    IO_FUNCTIONS.get_or_init(build_io_functions).as_slice()
}

pub fn os_functions() -> &'static [BuiltinFunction] {
    OS_FUNCTIONS.get_or_init(build_os_functions).as_slice()
}

pub fn rx_functions() -> &'static [BuiltinFunction] {
    RX_FUNCTIONS.get_or_init(build_rx_functions).as_slice()
}

pub fn test_functions() -> &'static [BuiltinFunction] {
    TEST_FUNCTIONS.get_or_init(build_test_functions).as_slice()
}

pub fn random_functions() -> &'static [BuiltinFunction] {
    RANDOM_FUNCTIONS.get_or_init(build_random_functions).as_slice()
}

fn build_random_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "random.int_effect",
            "Generate random integer in range [min, max) returning an Effect",
            &["min", "max"],
            vec![TypeExpr::Int, TypeExpr::Int],
            TypeExpr::Unknown,
        ),
        func(
            "random.float_effect",
            "Generate random float in range [0.0, 1.0) returning an Effect",
            &[],
            vec![],
            TypeExpr::Unknown,
        ),
        func(
            "random.bool_effect",
            "Generate random boolean returning an Effect",
            &[],
            vec![],
            TypeExpr::Unknown,
        ),
        func(
            "random.bytes_effect",
            "Generate random byte buffer of length len returning an Effect",
            &["len"],
            vec![TypeExpr::Int],
            TypeExpr::Unknown,
        ),
    ]
}

fn build_rx_functions() -> Vec<BuiltinFunction> {
    vec![
        func(
            "rx.Subject",
            "Create a new reactive Subject",
            &[],
            vec![],
            TypeExpr::Unknown,
        ),
        func(
            "rx.Observable",
            "Create an Observable from an array or source",
            &["source"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.from_iter",
            "Create an Observable from an array or iterable",
            &["items"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.interval",
            "Create a periodic timer Observable",
            &["period_ms"],
            vec![TypeExpr::Int],
            TypeExpr::Unknown,
        ),
        func(
            "rx.Scheduler",
            "Create a virtual reactive scheduler",
            &[],
            vec![],
            TypeExpr::Unknown,
        ),
        func(
            "rx.advance_by",
            "Advance virtual scheduler by delta milliseconds",
            &["sched", "delta_ms"],
            vec![TypeExpr::Unknown, TypeExpr::Int],
            TypeExpr::Unknown,
        ),
        func(
            "rx.unsubscribe",
            "Unsubscribe a reactive subscription",
            &["sub"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.join",
            "Join a reactive subscription awaiting completion",
            &["sub"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.BoundedQueue",
            "Create a dual-limit bounded queue with overflow strategy",
            &["max_items", "max_bytes", "strategy"],
            vec![TypeExpr::Int, TypeExpr::Int, TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "rx.BufferOptions",
            "Configure bounded buffer options",
            &["max_items", "max_bytes", "low_watermark_items", "low_watermark_bytes", "strategy"],
            vec![TypeExpr::Int, TypeExpr::Int, TypeExpr::Int, TypeExpr::Int, TypeExpr::String],
            TypeExpr::Unknown,
        ),
        func(
            "rx.default_buffer_options",
            "Default bounded buffer options (64 items / 64KB)",
            &[],
            vec![],
            TypeExpr::Unknown,
        ),
        func(
            "rx.push",
            "Push an item into bounded queue respecting overflow policy",
            &["queue", "item"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.pop",
            "Pop an item from bounded queue",
            &["queue"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.dequeue",
            "Dequeue an item from bounded queue",
            &["queue"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.dropped_count",
            "Get number of dropped items from bounded queue",
            &["queue"],
            vec![TypeExpr::Unknown],
            TypeExpr::Int,
        ),
        func(
            "rx.len",
            "Get number of items in queue or observable",
            &["target"],
            vec![TypeExpr::Unknown],
            TypeExpr::Int,
        ),
        func(
            "rx.used_bytes",
            "Get byte usage in bounded queue",
            &["queue"],
            vec![TypeExpr::Unknown],
            TypeExpr::Int,
        ),
        func(
            "rx.subject_subscribe",
            "Subscribe observer to Subject",
            &["subject", "next_fn"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.subject_subscribe_with",
            "Subscribe observer to Subject with callbacks and buffer options",
            &["subject", "next_fn", "err_fn", "complete_fn", "options"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown, TypeExpr::Unknown, TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.next",
            "Emit value to Subject subscribers",
            &["subject", "val"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.complete",
            "Complete Subject stream",
            &["subject"],
            vec![TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.filter",
            "Filter Observable items with predicate",
            &["obs", "pred_fn"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.map",
            "Transform Observable items with mapping function",
            &["obs", "map_fn"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.take",
            "Take first n items from Observable",
            &["obs", "n"],
            vec![TypeExpr::Unknown, TypeExpr::Int],
            TypeExpr::Unknown,
        ),
        func(
            "rx.subscribe",
            "Subscribe to Observable stream",
            &["obs", "next_fn"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
        func(
            "rx.subscribe_with",
            "Subscribe to Observable with error and completion callbacks",
            &["obs", "next_fn", "err_fn", "complete_fn"],
            vec![TypeExpr::Unknown, TypeExpr::Unknown, TypeExpr::Unknown, TypeExpr::Unknown],
            TypeExpr::Unknown,
        ),
    ]
}

fn build_test_functions() -> Vec<BuiltinFunction> {
    vec![
        func("test.clock_new", "Create a virtual TestClock", &[], vec![], TypeExpr::Unknown),
        func("test.clock_now_ms", "Get current virtual time in ms", &["clock"], vec![TypeExpr::Unknown], TypeExpr::Int),
        func("test.clock_schedule", "Schedule a fiber wake timer", &["clock", "fiber_id", "delay_ms"], vec![TypeExpr::Unknown, TypeExpr::Int, TypeExpr::Int], TypeExpr::Int),
        func("test.clock_adjust", "Fast-forward virtual time", &["clock", "delta_ms"], vec![TypeExpr::Unknown, TypeExpr::Int], TypeExpr::Unknown),
        func("test.clock_pending_count", "Pending timer count", &["clock"], vec![TypeExpr::Unknown], TypeExpr::Int),
        func("test.schedule_exponential", "Create exponential schedule", &["initial_ms", "factor", "max_delay_ms"], vec![TypeExpr::Int, TypeExpr::Float, TypeExpr::Int], TypeExpr::Unknown),
        func("test.schedule_step", "Step a schedule state", &["schedule", "state"], vec![TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.mock_new", "Create a mock service", &["id"], vec![TypeExpr::Int], TypeExpr::Unknown),
        func("test.mock_set_response", "Set mock response", &["mock", "endpoint", "response"], vec![TypeExpr::Unknown, TypeExpr::String, TypeExpr::String], TypeExpr::Unknown),
        func("test.mock_record_call", "Record mock service call", &["mock", "method", "args", "latency_ms"], vec![TypeExpr::Unknown, TypeExpr::String, TypeExpr::Unknown, TypeExpr::Int], TypeExpr::Unknown),
        func("test.mock_call_count", "Get mock call count", &["mock", "method"], vec![TypeExpr::Unknown, TypeExpr::String], TypeExpr::Int),
        func("test.mock_last_call", "Get mock last call record", &["mock", "method"], vec![TypeExpr::Unknown, TypeExpr::String], TypeExpr::Unknown),
        func("test.projection_new", "Create stack context projection", &["types", "services"], vec![TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.projection_get", "Get service from projection", &["projection", "type_id"], vec![TypeExpr::Unknown, TypeExpr::Int], TypeExpr::Unknown),
        func("test.resource_new", "Create mock host resource", &["name"], vec![TypeExpr::String], TypeExpr::Unknown),
        func("test.resource_is_active", "Check if host resource is active", &["resource"], vec![TypeExpr::Unknown], TypeExpr::Bool),
        func("test.scope_new", "Create a scope", &["id"], vec![TypeExpr::Int], TypeExpr::Unknown),
        func("test.scope_is_open", "Check if scope is open", &["scope"], vec![TypeExpr::Unknown], TypeExpr::Bool),
        func("test.scope_teardown_state", "Get scope teardown state", &["scope"], vec![TypeExpr::Unknown], TypeExpr::String),
        func("test.scope_admission_state", "Get scope admission state", &["scope"], vec![TypeExpr::Unknown], TypeExpr::String),
        func("test.scope_register_finalizer", "Register finalizer in scope", &["scope", "resource"], vec![TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.scope_register_child", "Register child fiber in scope", &["scope", "child"], vec![TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.scope_begin_teardown", "Begin scope teardown", &["scope"], vec![TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.scope_mark_joining", "Mark scope joining", &["scope"], vec![TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.scope_current_releases", "Get currently released resources", &["scope"], vec![TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.scope_finalize", "Finalize scope and run LIFO finalizers", &["scope"], vec![TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.fiber_new", "Create a fiber descriptor", &["id", "parent_id"], vec![TypeExpr::Int, TypeExpr::Int], TypeExpr::Unknown),
        func("test.fiber_set_state", "Set fiber execution state", &["fiber", "state"], vec![TypeExpr::Unknown, TypeExpr::String], TypeExpr::Unit),
        func("test.fiber_enter_mask", "Mask fiber interrupts", &["fiber"], vec![TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.fiber_exit_mask", "Unmask fiber interrupts", &["fiber"], vec![TypeExpr::Unknown], TypeExpr::Unknown),
        func("test.fiber_is_masked", "Check if fiber is masked", &["fiber"], vec![TypeExpr::Unknown], TypeExpr::Bool),
        func("test.fiber_request_interrupt", "Request fiber interrupt", &["fiber", "reason"], vec![TypeExpr::Unknown, TypeExpr::String], TypeExpr::Unknown),
        func("test.fiber_state", "Get fiber state", &["fiber"], vec![TypeExpr::Unknown], TypeExpr::String),
    ]
}

pub fn builtin_methods() -> &'static [BuiltinMethod] {
    BUILTIN_METHODS
        .get_or_init(build_builtin_methods)
        .as_slice()
}

pub struct BuiltinModule {
    name: &'static str,
    description: &'static str,
    functions: Vec<&'static BuiltinFunction>,
}

impl BuiltinModule {
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn description(&self) -> &'static str {
        self.description
    }

    pub fn functions(&self) -> &[&'static BuiltinFunction] {
        &self.functions
    }
}

pub struct BuiltinsDatabase {
    global_functions: Vec<&'static BuiltinFunction>,
    modules: BTreeMap<&'static str, BuiltinModule>,
    methods: HashMap<&'static str, Vec<&'static BuiltinMethod>>,
}

impl BuiltinsDatabase {
    pub fn global_functions(&self) -> &[&'static BuiltinFunction] {
        &self.global_functions
    }

    pub fn module(&self, name: &str) -> Option<&BuiltinModule> {
        self.modules.get(name)
    }

    pub fn methods_for(&self, type_name: &str) -> Option<&[&'static BuiltinMethod]> {
        self.methods
            .get(type_name)
            .map(|methods| methods.as_slice())
    }

    pub fn modules(&self) -> impl Iterator<Item = &BuiltinModule> {
        self.modules.values()
    }
}

fn receiver_key(expr: &TypeExpr) -> Option<&'static str> {
    match expr {
        TypeExpr::String => Some("String"),
        TypeExpr::Array(_) => Some("Array"),
        TypeExpr::Map(_, _) => Some("Map"),
        TypeExpr::Named(name) => Some(name),
        TypeExpr::Option(_) => Some("Option"),
        TypeExpr::Result(_, _) => Some("Result"),
        TypeExpr::Float => Some("Float"),
        TypeExpr::Int => Some("Int"),
        TypeExpr::Bool => Some("Bool"),
        TypeExpr::Unknown => Some("Unknown"),
        TypeExpr::Unit => Some("Unit"),
        TypeExpr::Generic(name) => Some(name),
        TypeExpr::SelfType => Some("Self"),
        TypeExpr::Function { .. } => Some("function"),
    }
}

static BUILTINS_DATABASE: StaticOnceCell<BuiltinsDatabase> = StaticOnceCell::new();

fn build_builtins_database() -> BuiltinsDatabase {
    let mut modules: BTreeMap<&'static str, BuiltinModule> = BTreeMap::new();
    let module_specs: [(&'static str, &'static str, &'static [BuiltinFunction]); 11] = [
        ("array", "array collection module", array_functions()),
        ("map", "map collection module", map_functions()),
        ("math", "math module", math_functions()),
        ("string", "string module", string_functions()),
        ("table", "table collection module", table_functions()),
        ("task", "task runtime module", task_functions()),
        ("io", "io file & console module", io_functions()),
        ("os", "os filesystem module", os_functions()),
        ("rx", "reactive streams module", rx_functions()),
        ("test", "test framework module", test_functions()),
        ("random", "pseudorandom number and byte generator module", random_functions()),
    ];
    for (name, description, functions) in module_specs {
        let mut module_funcs: Vec<&'static BuiltinFunction> = functions.iter().collect();
        module_funcs.sort_by(|a, b| a.name.cmp(b.name));
        modules.insert(
            name,
            BuiltinModule {
                name,
                description,
                functions: module_funcs,
            },
        );
    }

    let mut global_functions: Vec<&'static BuiltinFunction> = base_functions().iter().collect();
    global_functions.sort_by(|a, b| a.name.cmp(b.name));

    let mut methods: HashMap<&'static str, Vec<&'static BuiltinMethod>> = HashMap::new();
    for method in builtin_methods() {
        if let Some(key) = receiver_key(&method.receiver) {
            methods.entry(key).or_default().push(method);
        }
    }
    for vec in methods.values_mut() {
        vec.sort_by(|a, b| a.name.cmp(b.name));
    }

    BuiltinsDatabase {
        global_functions,
        modules,
        methods,
    }
}

pub fn builtins() -> &'static BuiltinsDatabase {
    BUILTINS_DATABASE.get_or_init(build_builtins_database)
}

pub fn lookup_builtin_method(
    receiver: &Type,
    name: &str,
) -> Option<(&'static BuiltinMethod, HashMap<&'static str, Type>)> {
    for method in builtin_methods() {
        if method.name == name
            && let Some(bindings) = match_receiver(&method.receiver, receiver)
        {
            return Some((method, bindings));
        }
    }
    None
}

pub fn is_virtual_std_module(module_path: &str) -> bool {
    let m = module_path.trim();
    let stripped = if let Some(s) = m.strip_prefix("@std/") {
        s
    } else if let Some(s) = m.strip_prefix("std.") {
        s
    } else {
        m
    };
    matches!(
        stripped,
        "math"
            | "string"
            | "array"
            | "map"
            | "table"
            | "io"
            | "os"
            | "task"
            | "lua"
            | "fmt"
            | "assert"
            | "iter"
            | "random"
            | "effect"
            | "schema"
            | "regex"
            | "data"
            | "model"
            | "test"
            | "test.clock"
            | "test.mock"
            | "bson"
            | "yaml"
            | "toon"
            | "json"
            | "jsonc"
            | "jsonl"
            | "rx"
    )
}

pub fn canonical_std_module_name(module_path: &str) -> &'static str {
    let m = module_path.trim();
    let stripped = if let Some(s) = m.strip_prefix("@std/") {
        s
    } else if let Some(s) = m.strip_prefix("std.") {
        s
    } else {
        m
    };
    match stripped {
        "math" => "math",
        "string" => "string",
        "array" => "array",
        "map" => "map",
        "table" => "table",
        "io" => "io",
        "os" => "os",
        "task" => "task",
        "lua" => "lua",
        "fmt" => "fmt",
        "assert" => "assert",
        "iter" => "iter",
        "random" => "random",
        "effect" => "effect",
        "schema" => "schema",
        "regex" => "regex",
        "data" => "data",
        "model" => "model",
        "bson" => "bson",
        "yaml" => "yaml",
        "toon" => "toon",
        "json" => "json",
        "jsonc" => "jsonc",
        "jsonl" => "jsonl",
        "test" | "test.clock" | "test.mock" => "test",
        "rx" => "rx",
        _ => "unknown",
    }
}

pub fn resolve_std_function(module_path: &str, item_name: &str) -> Option<&'static str> {
    let canon = canonical_std_module_name(module_path);
    match canon {
        "bson" => match item_name {
            "decode" => Some("bson.decode"),
            "encode" => Some("bson.encode"),
            "from_slice" => Some("bson.from_slice"),
            "to_vec" => Some("bson.to_vec"),
            "view" => Some("bson.view"),
            _ => None,
        },
        "yaml" => match item_name {
            "decode" => Some("yaml.decode"),
            "parse" => Some("yaml.parse"),
            "from_str" => Some("yaml.from_str"),
            "from_str_with_aliases" => Some("yaml.from_str_with_aliases"),
            "parse_documents" => Some("yaml.parse_documents"),
            "parse_documents_with_aliases" => Some("yaml.parse_documents_with_aliases"),
            _ => None,
        },
        "toon" => match item_name {
            "decode" => Some("toon.decode"),
            "encode" => Some("toon.encode"),
            "from_str" => Some("toon.from_str"),
            "to_string" => Some("toon.to_string"),
            "decode_table" => Some("toon.decode_table"),
            "encode_table" => Some("toon.encode_table"),
            _ => None,
        },
        "test" => match item_name {
            "test" => Some("std.test.test"),
            "suite" => Some("std.test.suite"),
            "describe" => Some("std.test.describe"),
            "assert" => Some("std.test.assert"),
            "assert_eq" => Some("std.test.assert_eq"),
            "assert_ne" => Some("std.test.assert_ne"),
            "assert_matches" => Some("std.test.assert_matches"),
            "assert_almost_eq" => Some("std.test.assert_almost_eq"),
            "assert_success" => Some("std.test.assert_success"),
            "assert_fails" => Some("std.test.assert_fails"),
            "assert_interrupted" => Some("std.test.assert_interrupted"),
            "assert_dies" => Some("std.test.assert_dies"),
            "before_each" => Some("std.test.before_each"),
            "after_each" => Some("std.test.after_each"),
            "before_all" => Some("std.test.before_all"),
            "after_all" => Some("std.test.after_all"),
            "clock_new" => Some("test.clock_new"),
            "clock_now_ms" => Some("test.clock_now_ms"),
            "clock_schedule" => Some("test.clock_schedule"),
            "clock_adjust" => Some("test.clock_adjust"),
            "clock_pending_count" => Some("test.clock_pending_count"),
            "schedule_exponential" => Some("test.schedule_exponential"),
            "schedule_step" => Some("test.schedule_step"),
            "mock_new" => Some("test.mock_new"),
            "mock_set_response" => Some("test.mock_set_response"),
            "mock_record_call" => Some("test.mock_record_call"),
            "mock_call_count" => Some("test.mock_call_count"),
            "mock_last_call" => Some("test.mock_last_call"),
            "projection_new" => Some("test.projection_new"),
            "projection_get" => Some("test.projection_get"),
            "resource_new" => Some("test.resource_new"),
            "resource_is_active" => Some("test.resource_is_active"),
            "scope_new" => Some("test.scope_new"),
            "scope_is_open" => Some("test.scope_is_open"),
            "scope_teardown_state" => Some("test.scope_teardown_state"),
            "scope_admission_state" => Some("test.scope_admission_state"),
            "scope_register_finalizer" => Some("test.scope_register_finalizer"),
            "scope_register_child" => Some("test.scope_register_child"),
            "scope_begin_teardown" => Some("test.scope_begin_teardown"),
            "scope_mark_joining" => Some("test.scope_mark_joining"),
            "scope_current_releases" => Some("test.scope_current_releases"),
            "scope_finalize" => Some("test.scope_finalize"),
            "fiber_new" => Some("test.fiber_new"),
            "fiber_set_state" => Some("test.fiber_set_state"),
            "fiber_enter_mask" => Some("test.fiber_enter_mask"),
            "fiber_exit_mask" => Some("test.fiber_exit_mask"),
            "fiber_is_masked" => Some("test.fiber_is_masked"),
            "fiber_request_interrupt" => Some("test.fiber_request_interrupt"),
            "fiber_state" => Some("test.fiber_state"),
            _ => None,
        },
        "effect" => match item_name {
            "succeed" => Some("effect.succeed"),
            "fail" => Some("effect.fail"),
            "die" => Some("effect.die"),
            "interrupt" => Some("effect.interrupt"),
            "from_result" => Some("effect.from_result"),
            "run_sync" => Some("effect.run_sync"),
            "suspend" => Some("effect.suspend"),
            "provide" => Some("effect.provide"),
            _ => None,
        },
        "schema" => match item_name {
            "s" => Some("schema.s"),
            _ => None,
        },
        "regex" => match item_name {
            "compile_bounded" => Some("regex.compile_bounded"),
            "is_match" => Some("regex.is_match"),
            "find" => Some("regex.find"),
            "replace_all" => Some("regex.replace_all"),
            _ => None,
        },
        "data" => match item_name {
            "document" => Some("data.document"),
            _ => None,
        },
        "model" => match item_name {
            "m" => Some("model.m"),
            _ => None,
        },
        "math" => match item_name {
            "abs" => Some("math.abs"),
            "floor" => Some("math.floor"),
            "ceil" => Some("math.ceil"),
            "round" => Some("math.round"),
            "sqrt" => Some("math.sqrt"),
            "sin" => Some("math.sin"),
            "cos" => Some("math.cos"),
            "tan" => Some("math.tan"),
            "asin" => Some("math.asin"),
            "acos" => Some("math.acos"),
            "atan" => Some("math.atan"),
            "atan2" => Some("math.atan2"),
            "min" => Some("math.min"),
            "max" => Some("math.max"),
            "clamp" => Some("math.clamp"),
            "random" => Some("math.random"),
            "randomseed" => Some("math.randomseed"),
            "deg" => Some("math.deg"),
            "rad" => Some("math.rad"),
            "exp" => Some("math.exp"),
            "log" => Some("math.log"),
            "tointeger" => Some("math.tointeger"),
            "tofloat" => Some("math.tofloat"),
            "mod" => Some("math.mod"),
            _ => None,
        },
        "io" => match item_name {
            "print" => Some("print"),
            "println" => Some("println"),
            "read_file" => Some("io.read_file"),
            "write_file" => Some("io.write_file"),
            "read_file_effect" => Some("io.read_file_effect"),
            "write_file_effect" => Some("io.write_file_effect"),
            "exists_effect" => Some("io.exists_effect"),
            "remove_file_effect" => Some("io.remove_file_effect"),
            "create_dir_all_effect" => Some("io.create_dir_all_effect"),
            "read_dir_effect" => Some("io.read_dir_effect"),
            "open_scoped" => Some("io.open_scoped"),
            "read_handle" => Some("io.read_handle"),
            "write_handle" => Some("io.write_handle"),
            "close_handle" => Some("io.close_handle"),
            _ => None,
        },
        "os" => match item_name {
            "time" => Some("os.time"),
            "date" => Some("os.date"),
            "sleep" => Some("os.sleep"),
            "create_file" => Some("os.create_file"),
            "create_dir" => Some("os.create_dir"),
            "remove_file" => Some("os.remove_file"),
            "remove_dir" => Some("os.remove_dir"),
            "rename" => Some("os.rename"),
            "get_env" => Some("os.get_env"),
            "getenv_effect" => Some("os.getenv_effect"),
            "setenv_effect" => Some("os.setenv_effect"),
            "now_ms_effect" => Some("os.now_ms_effect"),
            "args_effect" => Some("os.args_effect"),
            _ => None,
        },
        "fmt" => match item_name {
            "tostring" => Some("tostring"),
            "tonumber" => Some("tonumber"),
            "type" => Some("type"),
            _ => None,
        },
        "assert" => match item_name {
            "assert" => Some("assert"),
            "error" => Some("error"),
            _ => None,
        },
        "iter" => match item_name {
            "ipairs" => Some("ipairs"),
            "pairs" => Some("pairs"),
            _ => None,
        },
        "random" => match item_name {
            "random" => Some("random"),
            "randomseed" => Some("randomseed"),
            "int_effect" | "int" => Some("random.int_effect"),
            "float_effect" | "float" => Some("random.float_effect"),
            "bool_effect" | "bool" => Some("random.bool_effect"),
            "bytes_effect" | "bytes" => Some("random.bytes_effect"),
            _ => None,
        },
        "table" => match item_name {
            "len" => Some("table.len"),
            "is_empty" => Some("table.is_empty"),
            "get" => Some("table.get"),
            "first" => Some("table.first"),
            "last" => Some("table.last"),
            "push" => Some("table.push"),
            "pop" => Some("table.pop"),
            "insert" => Some("table.insert"),
            "remove" => Some("table.remove"),
            "clear" => Some("table.clear"),
            "slice" => Some("table.slice"),
            "concat" => Some("table.concat"),
            "sort" => Some("table.sort"),
            "reverse" => Some("table.reverse"),
            "contains" => Some("table.contains"),
            "map" => Some("table.map"),
            "filter" => Some("table.filter"),
            "reduce" => Some("table.reduce"),
            "iter" => Some("table.iter"),
            "unpack" => Some("table.unpack"),
            _ => None,
        },
        "string" => match item_name {
            "len" => Some("string.len"),
            "sub" => Some("string.sub"),
            "find" => Some("string.find"),
            "byte" => Some("string.byte"),
            "char" => Some("string.char"),
            "rep" => Some("string.rep"),
            "reverse" => Some("string.reverse"),
            "lower" => Some("string.lower"),
            "upper" => Some("string.upper"),
            "format" => Some("string.format"),
            "slice" => Some("string.slice"),
            "index_of" => Some("string.index_of"),
            _ => None,
        },
        "array" => match item_name {
            "len" => Some("array.len"),
            "push" => Some("array.push"),
            "pop" => Some("array.pop"),
            "get" => Some("array.get"),
            "set" => Some("array.set"),
            "is_empty" => Some("array.is_empty"),
            "first" => Some("array.first"),
            "last" => Some("array.last"),
            "insert" => Some("array.insert"),
            "remove" => Some("array.remove"),
            "clear" => Some("array.clear"),
            "slice" => Some("array.slice"),
            "concat" => Some("array.concat"),
            "sort" => Some("array.sort"),
            "reverse" => Some("array.reverse"),
            "contains" => Some("array.contains"),
            "map" => Some("array.map"),
            "filter" => Some("array.filter"),
            "reduce" => Some("array.reduce"),
            "iter" => Some("array.iter"),
            _ => None,
        },
        "map" => match item_name {
            "len" => Some("map.len"),
            "is_empty" => Some("map.is_empty"),
            "get" => Some("map.get"),
            "set" => Some("map.set"),
            "has" => Some("map.has"),
            "delete" => Some("map.delete"),
            "clear" => Some("map.clear"),
            "keys" => Some("map.keys"),
            "values" => Some("map.values"),
            "iter" => Some("map.iter"),
            _ => None,
        },
        "rx" => match item_name {
            "Subject" => Some("rx.Subject"),
            "Observable" => Some("rx.Observable"),
            "from_iter" => Some("rx.from_iter"),
            "interval" => Some("rx.interval"),
            "Scheduler" => Some("rx.Scheduler"),
            "advance_by" => Some("rx.advance_by"),
            "unsubscribe" => Some("rx.unsubscribe"),
            "join" => Some("rx.join"),
            "BoundedQueue" => Some("rx.BoundedQueue"),
            "BufferOptions" => Some("rx.BufferOptions"),
            "default_buffer_options" => Some("rx.default_buffer_options"),
            "push" => Some("rx.push"),
            "pop" => Some("rx.pop"),
            "dequeue" => Some("rx.dequeue"),
            "dropped_count" => Some("rx.dropped_count"),
            "len" => Some("rx.len"),
            "used_bytes" => Some("rx.used_bytes"),
            "subject_subscribe" => Some("rx.subject_subscribe"),
            "subject_subscribe_with" => Some("rx.subject_subscribe_with"),
            "next" => Some("rx.next"),
            "complete" => Some("rx.complete"),
            "filter" => Some("rx.filter"),
            "map" => Some("rx.map"),
            "take" => Some("rx.take"),
            "subscribe" => Some("rx.subscribe"),
            "subscribe_with" => Some("rx.subscribe_with"),
            _ => crate::rx::resolve_rx_function(item_name),
        },
        _ => None,
    }
}

pub fn virtual_std_module_functions(module_path: &str) -> Vec<(&'static str, &'static str)> {
    let canon = canonical_std_module_name(module_path);
    match canon {
        "effect" => vec![
            ("succeed", "effect.succeed"),
            ("fail", "effect.fail"),
            ("die", "effect.die"),
            ("interrupt", "effect.interrupt"),
            ("from_result", "effect.from_result"),
            ("run_sync", "effect.run_sync"),
            ("suspend", "effect.suspend"),
            ("provide", "effect.provide"),
        ],
        "math" => vec![
            ("abs", "math.abs"),
            ("floor", "math.floor"),
            ("ceil", "math.ceil"),
            ("round", "math.round"),
            ("sqrt", "math.sqrt"),
            ("sin", "math.sin"),
            ("cos", "math.cos"),
            ("tan", "math.tan"),
            ("asin", "math.asin"),
            ("acos", "math.acos"),
            ("atan", "math.atan"),
            ("atan2", "math.atan2"),
            ("min", "math.min"),
            ("max", "math.max"),
            ("clamp", "math.clamp"),
            ("random", "math.random"),
            ("randomseed", "math.randomseed"),
            ("deg", "math.deg"),
            ("rad", "math.rad"),
            ("exp", "math.exp"),
            ("log", "math.log"),
            ("tointeger", "math.tointeger"),
            ("tofloat", "math.tofloat"),
            ("mod", "math.mod"),
        ],
        "table" => vec![
            ("len", "table.len"),
            ("is_empty", "table.is_empty"),
            ("get", "table.get"),
            ("first", "table.first"),
            ("last", "table.last"),
            ("push", "table.push"),
            ("pop", "table.pop"),
            ("insert", "table.insert"),
            ("remove", "table.remove"),
            ("clear", "table.clear"),
            ("slice", "table.slice"),
            ("concat", "table.concat"),
            ("sort", "table.sort"),
            ("reverse", "table.reverse"),
            ("contains", "table.contains"),
            ("map", "table.map"),
            ("filter", "table.filter"),
            ("reduce", "table.reduce"),
            ("iter", "table.iter"),
            ("unpack", "table.unpack"),
        ],
        "string" => vec![
            ("len", "string.len"),
            ("sub", "string.sub"),
            ("find", "string.find"),
            ("byte", "string.byte"),
            ("char", "string.char"),
            ("rep", "string.rep"),
            ("reverse", "string.reverse"),
            ("lower", "string.lower"),
            ("upper", "string.upper"),
            ("format", "string.format"),
            ("slice", "string.slice"),
            ("index_of", "string.index_of"),
        ],
        "array" => vec![
            ("len", "array.len"),
            ("push", "array.push"),
            ("pop", "array.pop"),
            ("get", "array.get"),
            ("set", "array.set"),
            ("is_empty", "array.is_empty"),
            ("first", "array.first"),
            ("last", "array.last"),
            ("insert", "array.insert"),
            ("remove", "array.remove"),
            ("clear", "array.clear"),
            ("slice", "array.slice"),
            ("concat", "array.concat"),
            ("sort", "array.sort"),
            ("reverse", "array.reverse"),
            ("contains", "array.contains"),
            ("map", "array.map"),
            ("filter", "array.filter"),
            ("reduce", "array.reduce"),
            ("iter", "array.iter"),
        ],
        "map" => vec![
            ("len", "map.len"),
            ("is_empty", "map.is_empty"),
            ("get", "map.get"),
            ("set", "map.set"),
            ("has", "map.has"),
            ("delete", "map.delete"),
            ("clear", "map.clear"),
            ("keys", "map.keys"),
            ("values", "map.values"),
            ("iter", "map.iter"),
        ],
        "io" => vec![
            ("print", "print"),
            ("println", "println"),
            ("read_file", "io.read_file"),
            ("write_file", "io.write_file"),
            ("read_file_effect", "io.read_file_effect"),
            ("write_file_effect", "io.write_file_effect"),
            ("exists_effect", "io.exists_effect"),
            ("remove_file_effect", "io.remove_file_effect"),
            ("create_dir_all_effect", "io.create_dir_all_effect"),
            ("read_dir_effect", "io.read_dir_effect"),
            ("open_scoped", "io.open_scoped"),
            ("read_handle", "io.read_handle"),
            ("write_handle", "io.write_handle"),
            ("close_handle", "io.close_handle"),
        ],
        "os" => vec![
            ("time", "os.time"),
            ("date", "os.date"),
            ("sleep", "os.sleep"),
            ("create_file", "os.create_file"),
            ("create_dir", "os.create_dir"),
            ("remove_file", "os.remove_file"),
            ("remove_dir", "os.remove_dir"),
            ("rename", "os.rename"),
            ("get_env", "os.get_env"),
            ("getenv_effect", "os.getenv_effect"),
            ("setenv_effect", "os.setenv_effect"),
            ("now_ms_effect", "os.now_ms_effect"),
            ("args_effect", "os.args_effect"),
        ],
        "fmt" => vec![
            ("tostring", "tostring"),
            ("tonumber", "tonumber"),
            ("type", "type"),
        ],
        "assert" => vec![
            ("assert", "assert"),
            ("error", "error"),
        ],
        "iter" => vec![
            ("ipairs", "ipairs"),
            ("pairs", "pairs"),
        ],
        "random" => vec![
            ("random", "random"),
            ("randomseed", "randomseed"),
            ("int_effect", "random.int_effect"),
            ("float_effect", "random.float_effect"),
            ("bool_effect", "random.bool_effect"),
            ("bytes_effect", "random.bytes_effect"),
        ],
        "schema" => vec![
            ("s", "schema.s"),
        ],
        "regex" => vec![
            ("compile_bounded", "regex.compile_bounded"),
            ("is_match", "regex.is_match"),
            ("find", "regex.find"),
            ("replace_all", "regex.replace_all"),
        ],
        "rx" => vec![
            ("Subject", "rx.Subject"),
            ("Observable", "rx.Observable"),
            ("from_iter", "rx.from_iter"),
            ("interval", "rx.interval"),
            ("Scheduler", "rx.Scheduler"),
            ("advance_by", "rx.advance_by"),
            ("unsubscribe", "rx.unsubscribe"),
            ("join", "rx.join"),
            ("BoundedQueue", "rx.BoundedQueue"),
            ("BufferOptions", "rx.BufferOptions"),
            ("default_buffer_options", "rx.default_buffer_options"),
            ("push", "rx.push"),
            ("pop", "rx.pop"),
            ("dequeue", "rx.dequeue"),
            ("dropped_count", "rx.dropped_count"),
            ("len", "rx.len"),
            ("used_bytes", "rx.used_bytes"),
            ("subject_subscribe", "rx.subject_subscribe"),
            ("subject_subscribe_with", "rx.subject_subscribe_with"),
            ("next", "rx.next"),
            ("complete", "rx.complete"),
            ("filter", "rx.filter"),
            ("map", "rx.map"),
            ("take", "rx.take"),
            ("subscribe", "rx.subscribe"),
            ("subscribe_with", "rx.subscribe_with"),
        ],
        "test" => vec![
            ("assert", "std.test.assert"),
            ("assert_eq", "std.test.assert_eq"),
            ("assert_ne", "std.test.assert_ne"),
            ("assert_almost_eq", "std.test.assert_almost_eq"),
            ("assert_success", "std.test.assert_success"),
            ("assert_fails", "std.test.assert_fails"),
            ("assert_interrupted", "std.test.assert_interrupted"),
            ("assert_dies", "std.test.assert_dies"),
            ("clock_new", "test.clock_new"),
            ("clock_now_ms", "test.clock_now_ms"),
            ("clock_schedule", "test.clock_schedule"),
            ("clock_adjust", "test.clock_adjust"),
            ("clock_pending_count", "test.clock_pending_count"),
            ("schedule_exponential", "test.schedule_exponential"),
            ("schedule_step", "test.schedule_step"),
            ("mock_new", "test.mock_new"),
            ("mock_set_response", "test.mock_set_response"),
            ("mock_record_call", "test.mock_record_call"),
            ("mock_call_count", "test.mock_call_count"),
            ("mock_last_call", "test.mock_last_call"),
            ("projection_new", "test.projection_new"),
            ("projection_get", "test.projection_get"),
            ("resource_new", "test.resource_new"),
            ("resource_is_active", "test.resource_is_active"),
            ("scope_new", "test.scope_new"),
            ("scope_is_open", "test.scope_is_open"),
            ("scope_teardown_state", "test.scope_teardown_state"),
            ("scope_admission_state", "test.scope_admission_state"),
            ("scope_register_finalizer", "test.scope_register_finalizer"),
            ("scope_register_child", "test.scope_register_child"),
            ("scope_begin_teardown", "test.scope_begin_teardown"),
            ("scope_mark_joining", "test.scope_mark_joining"),
            ("scope_current_releases", "test.scope_current_releases"),
            ("scope_finalize", "test.scope_finalize"),
            ("fiber_new", "test.fiber_new"),
            ("fiber_set_state", "test.fiber_set_state"),
            ("fiber_enter_mask", "test.fiber_enter_mask"),
            ("fiber_exit_mask", "test.fiber_exit_mask"),
            ("fiber_is_masked", "test.fiber_is_masked"),
            ("fiber_request_interrupt", "test.fiber_request_interrupt"),
            ("fiber_state", "test.fiber_state"),
        ],
        _ => Vec::new(),
    }
}

pub fn find_builtin_function(name: &str) -> Option<&'static BuiltinFunction> {
    for func in base_functions() {
        if func.name == name {
            return Some(func);
        }
    }
    for func in math_functions() {
        if func.name == name || func.name.strip_prefix("math.") == Some(name) {
            return Some(func);
        }
    }
    for func in table_functions() {
        if func.name == name || func.name.strip_prefix("table.") == Some(name) {
            return Some(func);
        }
    }
    for func in string_functions() {
        if func.name == name || func.name.strip_prefix("string.") == Some(name) {
            return Some(func);
        }
    }
    for func in array_functions() {
        if func.name == name || func.name.strip_prefix("array.") == Some(name) {
            return Some(func);
        }
    }
    for func in map_functions() {
        if func.name == name || func.name.strip_prefix("map.") == Some(name) {
            return Some(func);
        }
    }
    for func in io_functions() {
        if func.name == name || func.name.strip_prefix("io.") == Some(name) {
            return Some(func);
        }
    }
    for func in os_functions() {
        if func.name == name || func.name.strip_prefix("os.") == Some(name) {
            return Some(func);
        }
    }
    for func in task_functions() {
        if func.name == name || func.name.strip_prefix("task.") == Some(name) {
            return Some(func);
        }
    }
    for func in lua_functions() {
        if func.name == name || func.name.strip_prefix("lua.") == Some(name) {
            return Some(func);
        }
    }
    for func in rx_functions() {
        if func.name == name || func.name.strip_prefix("rx.") == Some(name) {
            return Some(func);
        }
    }
    for func in test_functions() {
        if func.name == name || func.name.strip_prefix("test.") == Some(name) {
            return Some(func);
        }
    }
    for func in random_functions() {
        if func.name == name || func.name.strip_prefix("random.") == Some(name) {
            return Some(func);
        }
    }
    None
}

pub fn std_symbol_import_hint(name: &str) -> Option<&'static str> {
    match name {
        "print" => Some("help: import it with 'use std.io.{print};'"),
        "println" => Some("help: import it with 'use std.io.{println};'"),
        "assert" => Some("help: import it with 'use std.assert.{assert};'"),
        "error" => Some("help: import it with 'use std.assert.{error};'"),
        "tostring" => Some("help: import it with 'use std.fmt.{tostring};'"),
        "tonumber" => Some("help: import it with 'use std.fmt.{tonumber};'"),
        "type" => Some("help: import it with 'use std.fmt.{type};'"),
        "ipairs" => Some("help: import it with 'use std.iter.{ipairs};'"),
        "pairs" => Some("help: import it with 'use std.iter.{pairs};'"),
        "random" => Some("help: import it with 'use std.random.{random};'"),
        "randomseed" => Some("help: import it with 'use std.random.{randomseed};'"),
        "unpack" => Some("help: import it with 'use std.table.{unpack};'"),
        "select" => Some("help: import it with 'use std.io.{select};'"),
        "math" => Some("help: import it with 'use std.math;'"),
        "table" => Some("help: import it with 'use std.table;'"),
        "string" => Some("help: import it with 'use std.string;'"),
        "array" => Some("help: import it with 'use std.array;'"),
        "map" => Some("help: import it with 'use std.map;'"),
        "io" => Some("help: import it with 'use std.io;'"),
        "os" => Some("help: import it with 'use std.os;'"),
        "task" => Some("help: import it with 'use std.task;'"),
        "lua" => Some("help: import it with 'use std.lua;'"),
        _ => {
            if math_functions()
                .iter()
                .any(|f| f.name.strip_prefix("math.") == Some(name))
            {
                return Some("help: import it with 'use std.math;' or 'use std.math.{...};'");
            }
            if string_functions()
                .iter()
                .any(|f| f.name.strip_prefix("string.") == Some(name))
            {
                return Some("help: import it with 'use std.string;' or 'use std.string.{...};'");
            }
            if table_functions()
                .iter()
                .any(|f| f.name.strip_prefix("table.") == Some(name))
            {
                return Some("help: import it with 'use std.table;' or 'use std.table.{...};'");
            }
            if array_functions()
                .iter()
                .any(|f| f.name.strip_prefix("array.") == Some(name))
            {
                return Some("help: import it with 'use std.array;' or 'use std.array.{...};'");
            }
            if map_functions()
                .iter()
                .any(|f| f.name.strip_prefix("map.") == Some(name))
            {
                return Some("help: import it with 'use std.map;' or 'use std.map.{...};'");
            }
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_functions_registered() {
        let funcs = table_functions();
        assert_eq!(funcs.len(), 15);
        let names: Vec<&str> = funcs.iter().map(|f| f.name).collect();
        assert!(names.contains(&"table.len"));
        assert!(names.contains(&"table.is_empty"));
        assert!(names.contains(&"table.get"));
        assert!(names.contains(&"table.first"));
        assert!(names.contains(&"table.last"));
        assert!(names.contains(&"table.push"));
        assert!(names.contains(&"table.pop"));
        assert!(names.contains(&"table.insert"));
        assert!(names.contains(&"table.remove"));
        assert!(names.contains(&"table.concat"));
        assert!(names.contains(&"table.sort"));
        assert!(names.contains(&"table.reverse"));
        assert!(names.contains(&"table.clear"));
        assert!(names.contains(&"table.contains"));
        assert!(names.contains(&"table.slice"));
    }

    #[test]
    fn test_table_module_in_database() {
        let db = builtins();
        let table_mod = db.module("table").expect("table module must exist in builtins");
        assert_eq!(table_mod.functions().len(), 15);
    }

    #[test]
    fn test_math_mod_registered() {
        let funcs = math_functions();
        let math_mod = funcs.iter().find(|f| f.name == "math.mod");
        assert!(math_mod.is_some(), "math.mod must be registered");
        let sig = &math_mod.unwrap().signature;
        assert_eq!(sig.params.len(), 2);
    }

    #[test]
    fn test_string_slice_and_index_of_registered() {
        let funcs = string_functions();
        let slice_fn = funcs.iter().find(|f| f.name == "string.slice");
        assert!(slice_fn.is_some(), "string.slice must be registered");
        assert_eq!(slice_fn.unwrap().signature.params.len(), 3);

        let index_of_fn = funcs.iter().find(|f| f.name == "string.index_of");
        assert!(index_of_fn.is_some(), "string.index_of must be registered");
        assert_eq!(index_of_fn.unwrap().signature.params.len(), 2);
    }
}
