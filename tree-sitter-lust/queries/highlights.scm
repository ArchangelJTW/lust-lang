; Keywords
[
  "function"
  "local"
  "if"
  "then"
  "else"
  "elseif"
  "end"
  "while"
  "do"
  "for"
  "return"
  "struct"
  "impl"
  "enum"
  "match"
  "case"
  "trait"
  "use"
  "as"
  "is"
] @keyword

; Control flow keywords
[
  "if"
  "then"
  "else"
  "elseif"
  "match"
  "case"
] @keyword.control.conditional

[
  "while"
  "for"
  "do"
] @keyword.control.repeat

; Break and continue statements
(break_statement) @keyword.control.repeat
(continue_statement) @keyword.control.repeat

[
  "return"
] @keyword.control.return

; Function keywords
"function" @keyword.function

; Type keywords
[
  "struct"
  "enum"
  "trait"
  "impl"
] @keyword.type

; Storage keywords
"local" @keyword.storage

; Operators
[
  "+"
  "-"
  "*"
  "/"
  "%"
  "=="
  "!="
  "<"
  "<="
  ">"
  ">="
  "and"
  "or"
  "not"
  "="
  "+="
  "-="
  "*="
  "/="
  "%="
  ".."
] @operator

; Special operators
":" @punctuation.delimiter
"." @punctuation.delimiter
"is" @keyword.operator

; Delimiters
[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

[
  ","
] @punctuation.delimiter

; Function declarations
(function_declaration
  name: (identifier) @function)

(trait_method
  name: (identifier) @function)

; Function calls
(call_expression
  function: (identifier) @function.call)

; Method calls
(method_call_expression
  method: (identifier) @function.method.call)

; Struct declarations
(struct_declaration
  name: (identifier) @type)

(struct_field
  name: (identifier) @variable.member)

; Enum declarations
(enum_declaration
  name: (identifier) @type)

(enum_variant
  name: (identifier) @constructor)

; Trait declarations
(trait_declaration
  name: (identifier) @type)

; Impl blocks
(impl_block
  type: (identifier) @type)

; Type annotations
(type_annotation
  (identifier) @type)

(primitive_type) @type.builtin

; Struct expressions
(struct_expression
  type: (identifier) @constructor)

(struct_field_init
  name: (identifier) @variable.member)

(map_entry
  key: (identifier) @variable.member)

; Field access
(field_access
  field: (identifier) @variable.member)

; Parameters
(parameter
  name: (identifier) @variable.parameter)

; Local declarations
(binding
  name: (identifier) @variable)

; Variables
(identifier) @variable

; Literals
(number) @number
(string) @string
(boolean) @boolean
(nil) @constant.builtin

; Comments
(comment) @comment

; Type parameters
(type_parameters
  (identifier) @type.parameter)

; Pattern matching
(pattern
  (identifier) @variable)

(enum_pattern
  variant: (identifier) @constructor)

; Special identifiers
((identifier) @constant
 (#match? @constant "^[A-Z][A-Z0-9_]*$"))

; Built-in types
[
  "int"
  "float"
  "bool"
  "string"
  "nil"
  "unknown"
] @type.builtin

; Built-in functions
((identifier) @function.builtin
 (#match? @function.builtin "^(print|println|type|tostring)$"))
