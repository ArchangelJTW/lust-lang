use crate::ast::{Span, Type, TypeKind};
use crate::lazy::StaticOnceCell;
use crate::FunctionSignature;
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
static MAP_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static MATH_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static STRING_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static TASK_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static LUA_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static IO_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();
static OS_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();

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
        func("math.abs", "Return the absolute value of a number", &["x"], vec![TypeExpr::Unknown], TypeExpr::Unknown),
        func("math.floor", "Return the largest integer less than or equal to x", &["x"], vec![TypeExpr::Unknown], TypeExpr::Int),
        func("math.ceil", "Return the smallest integer greater than or equal to x", &["x"], vec![TypeExpr::Unknown], TypeExpr::Int),
        func("math.round", "Round to the nearest integer", &["x"], vec![TypeExpr::Unknown], TypeExpr::Unknown),
        func("math.sqrt", "Return the square root of x", &["x"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.sin", "Return the sine of x in radians", &["x"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.cos", "Return the cosine of x in radians", &["x"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.tan", "Return the tangent of x in radians", &["x"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.asin", "Return the arcsine of x in radians", &["x"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.acos", "Return the arccosine of x in radians", &["x"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.atan", "Return the arctangent of x in radians", &["x"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.atan2", "Return the arctangent of y/x in radians", &["y", "x"], vec![TypeExpr::Float, TypeExpr::Float], TypeExpr::Float),
        func("math.min", "Return the minimum of the provided numbers", &["x", "y"], vec![TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("math.max", "Return the maximum of the provided numbers", &["x", "y"], vec![TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("math.clamp", "Clamp a number between min and max bounds", &["x", "min", "max"], vec![TypeExpr::Unknown, TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("math.random", "Generate a pseudo-random number", &["m", "n"], vec![TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("math.randomseed", "Set the seed for the pseudo-random number generator", &["seed"], vec![TypeExpr::Unknown], TypeExpr::Unit),
        func("math.deg", "Convert angle from radians to degrees", &["rad"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.rad", "Convert angle from degrees to radians", &["deg"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.exp", "Return e^x", &["x"], vec![TypeExpr::Float], TypeExpr::Float),
        func("math.log", "Return the natural logarithm of x (or logarithm to optional base)", &["x", "base"], vec![TypeExpr::Float, TypeExpr::Unknown], TypeExpr::Float),
        func("math.tointeger", "Convert a number to an integer if possible", &["x"], vec![TypeExpr::Unknown], TypeExpr::Option(Box::new(TypeExpr::Int))),
        func("math.tofloat", "Convert a number to a float", &["x"], vec![TypeExpr::Unknown], TypeExpr::Option(Box::new(TypeExpr::Float))),
    ]
}

fn build_string_functions() -> Vec<BuiltinFunction> {
    vec![
        func("string.len", "Return the length of a string in bytes", &["s"], vec![TypeExpr::String], TypeExpr::Int),
        func("string.sub", "Extract a substring using 1-based indices", &["s", "i", "j"], vec![TypeExpr::String, TypeExpr::Int, TypeExpr::Unknown], TypeExpr::String),
        func("string.lower", "Convert a string to lowercase", &["s"], vec![TypeExpr::String], TypeExpr::String),
        func("string.upper", "Convert a string to uppercase", &["s"], vec![TypeExpr::String], TypeExpr::String),
        func("string.byte", "Return internal numeric codes of characters", &["s", "i", "j"], vec![TypeExpr::String, TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("string.char", "Create a string from integer character codes", &["byte"], vec![TypeExpr::Unknown], TypeExpr::String),
        func("string.find", "Find the first match of pattern in string", &["s", "pattern", "init", "plain"], vec![TypeExpr::String, TypeExpr::String, TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("string.match", "Extract pattern captures from string", &["s", "pattern", "init"], vec![TypeExpr::String, TypeExpr::String, TypeExpr::Unknown], TypeExpr::Unknown),
        func("string.gsub", "Global pattern substitution in string", &["s", "pattern", "repl", "n"], vec![TypeExpr::String, TypeExpr::String, TypeExpr::Unknown, TypeExpr::Unknown], TypeExpr::Unknown),
        func("string.format", "Format a string using printf-style specifiers", &["fmt", "..."], vec![TypeExpr::String, TypeExpr::Unknown], TypeExpr::String),
        func("string.rep", "Return a string repeated n times", &["s", "n", "sep"], vec![TypeExpr::String, TypeExpr::Int, TypeExpr::Unknown], TypeExpr::String),
        func("string.reverse", "Return a string with reversed characters", &["s"], vec![TypeExpr::String], TypeExpr::String),
        func("string.split", "Split a string by delimiter", &["s", "delimiter"], vec![TypeExpr::String, TypeExpr::String], TypeExpr::Array(Box::new(TypeExpr::String))),
        func("string.trim", "Trim whitespace from start and end of string", &["s"], vec![TypeExpr::String], TypeExpr::String),
        func("string.trim_start", "Trim leading whitespace from string", &["s"], vec![TypeExpr::String], TypeExpr::String),
        func("string.trim_end", "Trim trailing whitespace from string", &["s"], vec![TypeExpr::String], TypeExpr::String),
        func("string.replace", "Replace occurrences of substring with replacement", &["s", "from", "to"], vec![TypeExpr::String, TypeExpr::String, TypeExpr::String], TypeExpr::String),
        func("string.starts_with", "Check if string starts with prefix", &["s", "prefix"], vec![TypeExpr::String, TypeExpr::String], TypeExpr::Bool),
        func("string.ends_with", "Check if string ends with suffix", &["s", "suffix"], vec![TypeExpr::String, TypeExpr::String], TypeExpr::Bool),
        func("string.contains", "Check if string contains substring", &["s", "substring"], vec![TypeExpr::String, TypeExpr::String], TypeExpr::Bool),
        func("string.is_empty", "Check if string is empty", &["s"], vec![TypeExpr::String], TypeExpr::Bool),
        func("string.chars", "Return characters of string as array of single-character strings", &["s"], vec![TypeExpr::String], TypeExpr::Array(Box::new(TypeExpr::String))),
        func("string.lines", "Return lines of string as array of strings", &["s"], vec![TypeExpr::String], TypeExpr::Array(Box::new(TypeExpr::String))),
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
            TypeExpr::Result(
                Box::new(TypeExpr::String),
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
            TypeExpr::Result(
                Box::new(TypeExpr::String),
                Box::new(TypeExpr::String),
            ),
        ),
        func(
            "io.read_line",
            "Read a single line from stdin",
            &[],
            vec![],
            TypeExpr::Result(
                Box::new(TypeExpr::String),
                Box::new(TypeExpr::String),
            ),
        ),
        func(
            "io.write_stdout",
            "Write a value to stdout",
            &["value"],
            vec![TypeExpr::Unknown],
            TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
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
    let module_specs: [(&'static str, &'static str, &'static [BuiltinFunction]); 7] = [
        ("array", "array collection module", array_functions()),
        ("map", "map collection module", map_functions()),
        ("math", "math module", math_functions()),
        ("string", "string module", string_functions()),
        ("task", "task runtime module", task_functions()),
        ("io", "io file & console module", io_functions()),
        ("os", "os filesystem module", os_functions()),
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
        if method.name == name {
            if let Some(bindings) = match_receiver(&method.receiver, receiver) {
                return Some((method, bindings));
            }
        }
    }
    None
}
