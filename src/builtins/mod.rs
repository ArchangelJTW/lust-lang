use crate::ast::{Span, Type, TypeKind};
use crate::lazy::StaticOnceCell;
use crate::FunctionSignature;
use alloc::{boxed::Box, collections::BTreeMap, string::ToString, vec, vec::Vec};
use hashbrown::HashMap;

#[derive(Debug, Clone)]
pub struct BuiltinSignature {
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
                .unwrap_or_else(|| Type::new(TypeKind::Unknown, span)),
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
            params,
            return_type,
        },
        param_names,
        semantics: MethodSemantics::Simple,
    }
}

fn method_with_semantics(
    receiver: TypeExpr,
    name: &'static str,
    description: &'static str,
    param_names: &'static [&'static str],
    params: Vec<TypeExpr>,
    return_type: TypeExpr,
    semantics: MethodSemantics,
) -> BuiltinMethod {
    let mut m = method(
        receiver,
        name,
        description,
        param_names,
        params,
        return_type,
    );
    m.semantics = semantics;
    m
}

fn string_methods() -> Vec<BuiltinMethod> {
    vec![
        method(
            TypeExpr::String,
            "len",
            "Return the length of the string in bytes",
            &[],
            vec![],
            TypeExpr::Int,
        ),
        method(
            TypeExpr::String,
            "substring",
            "Extract a substring from the string",
            &["start", "end"],
            vec![TypeExpr::Int, TypeExpr::Int],
            TypeExpr::String,
        ),
        method(
            TypeExpr::String,
            "find",
            "Find the first occurrence of a substring",
            &["pattern"],
            vec![TypeExpr::String],
            TypeExpr::Option(Box::new(TypeExpr::Int)),
        ),
        method(
            TypeExpr::String,
            "starts_with",
            "Check whether the string starts with a prefix",
            &["prefix"],
            vec![TypeExpr::String],
            TypeExpr::Bool,
        ),
        method(
            TypeExpr::String,
            "ends_with",
            "Check whether the string ends with a suffix",
            &["suffix"],
            vec![TypeExpr::String],
            TypeExpr::Bool,
        ),
        method(
            TypeExpr::String,
            "contains",
            "Check whether the string contains a substring",
            &["substring"],
            vec![TypeExpr::String],
            TypeExpr::Bool,
        ),
        method(
            TypeExpr::String,
            "split",
            "Split the string on a separator",
            &["delimiter"],
            vec![TypeExpr::String],
            TypeExpr::Array(Box::new(TypeExpr::String)),
        ),
        method(
            TypeExpr::String,
            "trim",
            "Trim whitespace from both ends of the string",
            &[],
            vec![],
            TypeExpr::String,
        ),
        method(
            TypeExpr::String,
            "trim_start",
            "Trim whitespace from the start of the string",
            &[],
            vec![],
            TypeExpr::String,
        ),
        method(
            TypeExpr::String,
            "trim_end",
            "Trim whitespace from the end of the string",
            &[],
            vec![],
            TypeExpr::String,
        ),
        method(
            TypeExpr::String,
            "replace",
            "Replace occurrences of a substring",
            &["from", "to"],
            vec![TypeExpr::String, TypeExpr::String],
            TypeExpr::String,
        ),
        method(
            TypeExpr::String,
            "to_upper",
            "Convert the string to uppercase",
            &[],
            vec![],
            TypeExpr::String,
        ),
        method(
            TypeExpr::String,
            "to_lower",
            "Convert the string to lowercase",
            &[],
            vec![],
            TypeExpr::String,
        ),
        method(
            TypeExpr::String,
            "is_empty",
            "Check if the string is empty",
            &[],
            vec![],
            TypeExpr::Bool,
        ),
        method(
            TypeExpr::String,
            "chars",
            "Return the characters as an array of strings",
            &[],
            vec![],
            TypeExpr::Array(Box::new(TypeExpr::String)),
        ),
        method(
            TypeExpr::String,
            "lines",
            "Return the lines as an array of strings",
            &[],
            vec![],
            TypeExpr::Array(Box::new(TypeExpr::String)),
        ),
        method(
            TypeExpr::String,
            "iter",
            "Return an iterator over the characters of the string",
            &[],
            vec![],
            TypeExpr::Named("Iterator"),
        ),
    ]
}

fn array_methods() -> Vec<BuiltinMethod> {
    let receiver = TypeExpr::Array(Box::new(TypeExpr::Generic("T")));
    let mut methods = Vec::new();
    methods.push(method(
        receiver.clone(),
        "iter",
        "Return an iterator over the array items",
        &[],
        vec![],
        TypeExpr::Named("Iterator"),
    ));
    methods.push(method(
        receiver.clone(),
        "len",
        "Return the number of elements in the array",
        &[],
        vec![],
        TypeExpr::Int,
    ));
    methods.push(method(
        receiver.clone(),
        "get",
        "Return the element at the given index, if any",
        &["index"],
        vec![TypeExpr::Int],
        TypeExpr::Option(Box::new(TypeExpr::Generic("T"))),
    ));
    methods.push(method(
        receiver.clone(),
        "first",
        "Return the first element, if any",
        &[],
        vec![],
        TypeExpr::Option(Box::new(TypeExpr::Generic("T"))),
    ));
    methods.push(method(
        receiver.clone(),
        "last",
        "Return the last element, if any",
        &[],
        vec![],
        TypeExpr::Option(Box::new(TypeExpr::Generic("T"))),
    ));
    methods.push(method(
        receiver.clone(),
        "push",
        "Append a value to the array",
        &["value"],
        vec![TypeExpr::Generic("T")],
        TypeExpr::Unit,
    ));
    methods.push(method(
        receiver.clone(),
        "pop",
        "Remove and return the last element, if any",
        &[],
        vec![],
        TypeExpr::Option(Box::new(TypeExpr::Generic("T"))),
    ));
    methods.push(method_with_semantics(
        receiver.clone(),
        "map",
        "Transform each element using the provided function",
        &["func"],
        vec![TypeExpr::Function {
            params: vec![TypeExpr::Generic("T")],
            return_type: Box::new(TypeExpr::Unknown),
        }],
        TypeExpr::Array(Box::new(TypeExpr::Unknown)),
        MethodSemantics::ArrayMap,
    ));
    methods.push(method_with_semantics(
        receiver.clone(),
        "filter",
        "Keep elements where the predicate returns true",
        &["func"],
        vec![TypeExpr::Function {
            params: vec![TypeExpr::Generic("T")],
            return_type: Box::new(TypeExpr::Bool),
        }],
        TypeExpr::Array(Box::new(TypeExpr::Generic("T"))),
        MethodSemantics::ArrayFilter,
    ));
    methods.push(method_with_semantics(
        receiver.clone(),
        "reduce",
        "Fold elements into a single value",
        &["initial", "func"],
        vec![
            TypeExpr::Unknown,
            TypeExpr::Function {
                params: vec![TypeExpr::Unknown, TypeExpr::Generic("T")],
                return_type: Box::new(TypeExpr::Unknown),
            },
        ],
        TypeExpr::Unknown,
        MethodSemantics::ArrayReduce,
    ));
    methods.push(method(
        receiver.clone(),
        "slice",
        "Return a slice of the array between two indices",
        &["start", "end"],
        vec![TypeExpr::Int, TypeExpr::Int],
        TypeExpr::Array(Box::new(TypeExpr::Generic("T"))),
    ));
    methods.push(method(
        receiver.clone(),
        "clear",
        "Remove all elements from the array",
        &[],
        vec![],
        TypeExpr::Unit,
    ));
    methods.push(method(
        receiver,
        "is_empty",
        "Check if the array contains no elements",
        &[],
        vec![],
        TypeExpr::Bool,
    ));
    methods
}

fn map_methods() -> Vec<BuiltinMethod> {
    let receiver = TypeExpr::Map(
        Box::new(TypeExpr::Generic("K")),
        Box::new(TypeExpr::Generic("V")),
    );
    vec![
        method(
            receiver.clone(),
            "iter",
            "Iterate over key/value pairs",
            &[],
            vec![],
            TypeExpr::Named("Iterator"),
        ),
        method(
            receiver.clone(),
            "len",
            "Return the number of entries in the map",
            &[],
            vec![],
            TypeExpr::Int,
        ),
        method(
            receiver.clone(),
            "get",
            "Look up a value by key",
            &["key"],
            vec![TypeExpr::Generic("K")],
            TypeExpr::Option(Box::new(TypeExpr::Generic("V"))),
        ),
        method(
            receiver.clone(),
            "set",
            "Insert or overwrite a key/value pair",
            &["key", "value"],
            vec![TypeExpr::Generic("K"), TypeExpr::Generic("V")],
            TypeExpr::Unit,
        ),
        method(
            receiver.clone(),
            "has",
            "Check whether the map contains a key",
            &["key"],
            vec![TypeExpr::Generic("K")],
            TypeExpr::Bool,
        ),
        method(
            receiver.clone(),
            "delete",
            "Remove an entry from the map",
            &["key"],
            vec![TypeExpr::Generic("K")],
            TypeExpr::Option(Box::new(TypeExpr::Generic("V"))),
        ),
        method(
            receiver.clone(),
            "keys",
            "Return the keys as an array",
            &[],
            vec![],
            TypeExpr::Array(Box::new(TypeExpr::Generic("K"))),
        ),
        method(
            receiver,
            "values",
            "Return the values as an array",
            &[],
            vec![],
            TypeExpr::Array(Box::new(TypeExpr::Generic("V"))),
        ),
    ]
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

fn float_methods() -> Vec<BuiltinMethod> {
    vec![
        method(
            TypeExpr::Float,
            "to_int",
            "Convert the float to an integer by truncation",
            &[],
            vec![],
            TypeExpr::Int,
        ),
        method(
            TypeExpr::Float,
            "floor",
            "Return the greatest integer less than or equal to the value",
            &[],
            vec![],
            TypeExpr::Float,
        ),
        method(
            TypeExpr::Float,
            "ceil",
            "Return the smallest integer greater than or equal to the value",
            &[],
            vec![],
            TypeExpr::Float,
        ),
        method(
            TypeExpr::Float,
            "round",
            "Round the float to the nearest integer",
            &[],
            vec![],
            TypeExpr::Float,
        ),
        method(
            TypeExpr::Float,
            "sqrt",
            "Return the square root of the float",
            &[],
            vec![],
            TypeExpr::Float,
        ),
        method(
            TypeExpr::Float,
            "abs",
            "Return the absolute value of the float",
            &[],
            vec![],
            TypeExpr::Float,
        ),
        method(
            TypeExpr::Float,
            "clamp",
            "Clamp the float between a minimum and maximum value",
            &["min", "max"],
            vec![TypeExpr::Float, TypeExpr::Float],
            TypeExpr::Float,
        ),
    ]
}

fn int_methods() -> Vec<BuiltinMethod> {
    vec![
        method(
            TypeExpr::Int,
            "to_float",
            "Convert the integer to a float",
            &[],
            vec![],
            TypeExpr::Float,
        ),
        method(
            TypeExpr::Int,
            "abs",
            "Return the absolute value of the integer",
            &[],
            vec![],
            TypeExpr::Int,
        ),
        method(
            TypeExpr::Int,
            "clamp",
            "Clamp the integer between a minimum and maximum value",
            &["min", "max"],
            vec![TypeExpr::Int, TypeExpr::Int],
            TypeExpr::Int,
        ),
    ]
}

static BASE_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();

fn build_base_functions() -> Vec<BuiltinFunction> {
    vec![
        BuiltinFunction {
            name: "print",
            description: "Print values without a newline",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Unknown],
                return_type: TypeExpr::Unit,
            },
            param_names: &["value"],
        },
        BuiltinFunction {
            name: "println",
            description: "Print values followed by a newline",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Unknown],
                return_type: TypeExpr::Unit,
            },
            param_names: &["value"],
        },
        BuiltinFunction {
            name: "type",
            description: "Return the runtime type name",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Unknown],
                return_type: TypeExpr::String,
            },
            param_names: &["value"],
        },
        BuiltinFunction {
            name: "tostring",
            description: "Convert a value to a string",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Unknown],
                return_type: TypeExpr::String,
            },
            param_names: &["value"],
        },
    ]
}

static TASK_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();

fn build_task_functions() -> Vec<BuiltinFunction> {
    vec![
        BuiltinFunction {
            name: "task.run",
            description: "Run a function as a task",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Unknown],
                return_type: TypeExpr::Named("Task"),
            },
            param_names: &["func"],
        },
        BuiltinFunction {
            name: "task.create",
            description: "Create a suspended task",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Unknown],
                return_type: TypeExpr::Named("Task"),
            },
            param_names: &["func"],
        },
        BuiltinFunction {
            name: "task.status",
            description: "Get the status of a task",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Named("Task")],
                return_type: TypeExpr::Named("TaskStatus"),
            },
            param_names: &["task"],
        },
        BuiltinFunction {
            name: "task.info",
            description: "Get detailed information about a task",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Named("Task")],
                return_type: TypeExpr::Named("TaskInfo"),
            },
            param_names: &["task"],
        },
        BuiltinFunction {
            name: "task.resume",
            description: "Resume a suspended task",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Named("Task")],
                return_type: TypeExpr::Named("TaskInfo"),
            },
            param_names: &["task"],
        },
        BuiltinFunction {
            name: "task.yield",
            description: "Yield from the current task",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Unknown],
                return_type: TypeExpr::Unknown,
            },
            param_names: &["value"],
        },
        BuiltinFunction {
            name: "task.stop",
            description: "Stop a running task",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Named("Task")],
                return_type: TypeExpr::Bool,
            },
            param_names: &["task"],
        },
        BuiltinFunction {
            name: "task.restart",
            description: "Restart a completed task",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Named("Task")],
                return_type: TypeExpr::Named("TaskInfo"),
            },
            param_names: &["task"],
        },
        BuiltinFunction {
            name: "task.current",
            description: "Return the currently executing task",
            signature: BuiltinSignature {
                params: vec![],
                return_type: TypeExpr::Option(Box::new(TypeExpr::Named("Task"))),
            },
            param_names: &[],
        },
    ]
}

static IO_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();

fn build_io_functions() -> Vec<BuiltinFunction> {
    vec![
        BuiltinFunction {
            name: "io.read_file",
            description: "Read the contents of a file",
            signature: BuiltinSignature {
                params: vec![TypeExpr::String],
                return_type: TypeExpr::Result(
                    Box::new(TypeExpr::String),
                    Box::new(TypeExpr::String),
                ),
            },
            param_names: &["path"],
        },
        BuiltinFunction {
            name: "io.read_file_bytes",
            description: "Read the contents of a file as byte values",
            signature: BuiltinSignature {
                params: vec![TypeExpr::String],
                return_type: TypeExpr::Result(
                    Box::new(TypeExpr::Array(Box::new(TypeExpr::Int))),
                    Box::new(TypeExpr::String),
                ),
            },
            param_names: &["path"],
        },
        BuiltinFunction {
            name: "io.write_file",
            description: "Write contents to a file",
            signature: BuiltinSignature {
                params: vec![TypeExpr::String, TypeExpr::Unknown],
                return_type: TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
            },
            param_names: &["path", "value"],
        },
        BuiltinFunction {
            name: "io.read_stdin",
            description: "Read all available stdin",
            signature: BuiltinSignature {
                params: vec![],
                return_type: TypeExpr::Result(
                    Box::new(TypeExpr::String),
                    Box::new(TypeExpr::String),
                ),
            },
            param_names: &[],
        },
        BuiltinFunction {
            name: "io.read_line",
            description: "Read a single line from stdin",
            signature: BuiltinSignature {
                params: vec![],
                return_type: TypeExpr::Result(
                    Box::new(TypeExpr::String),
                    Box::new(TypeExpr::String),
                ),
            },
            param_names: &[],
        },
        BuiltinFunction {
            name: "io.write_stdout",
            description: "Write a value to stdout",
            signature: BuiltinSignature {
                params: vec![TypeExpr::Unknown],
                return_type: TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
            },
            param_names: &["value"],
        },
    ]
}

static OS_FUNCTIONS: StaticOnceCell<Vec<BuiltinFunction>> = StaticOnceCell::new();

fn build_os_functions() -> Vec<BuiltinFunction> {
    vec![
        BuiltinFunction {
            name: "os.create_file",
            description: "Create an empty file on disk",
            signature: BuiltinSignature {
                params: vec![TypeExpr::String],
                return_type: TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
            },
            param_names: &["path"],
        },
        BuiltinFunction {
            name: "os.create_dir",
            description: "Create a directory",
            signature: BuiltinSignature {
                params: vec![TypeExpr::String],
                return_type: TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
            },
            param_names: &["path"],
        },
        BuiltinFunction {
            name: "os.remove_file",
            description: "Remove a file from disk",
            signature: BuiltinSignature {
                params: vec![TypeExpr::String],
                return_type: TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
            },
            param_names: &["path"],
        },
        BuiltinFunction {
            name: "os.remove_dir",
            description: "Remove an empty directory",
            signature: BuiltinSignature {
                params: vec![TypeExpr::String],
                return_type: TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
            },
            param_names: &["path"],
        },
        BuiltinFunction {
            name: "os.rename",
            description: "Rename or move a path",
            signature: BuiltinSignature {
                params: vec![TypeExpr::String, TypeExpr::String],
                return_type: TypeExpr::Result(Box::new(TypeExpr::Unit), Box::new(TypeExpr::String)),
            },
            param_names: &["from", "to"],
        },
    ]
}

static BUILTIN_METHODS: StaticOnceCell<Vec<BuiltinMethod>> = StaticOnceCell::new();

fn build_builtin_methods() -> Vec<BuiltinMethod> {
    let mut methods = Vec::new();
    methods.extend(string_methods());
    methods.extend(array_methods());
    methods.extend(map_methods());
    methods.extend(iterator_methods());
    methods.extend(option_methods());
    methods.extend(result_methods());
    methods.extend(float_methods());
    methods.extend(int_methods());
    methods
}

pub fn base_functions() -> &'static [BuiltinFunction] {
    BASE_FUNCTIONS.get_or_init(build_base_functions).as_slice()
}

pub fn task_functions() -> &'static [BuiltinFunction] {
    TASK_FUNCTIONS.get_or_init(build_task_functions).as_slice()
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
    let module_specs: [(&'static str, &'static str, &'static [BuiltinFunction]); 3] = [
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
