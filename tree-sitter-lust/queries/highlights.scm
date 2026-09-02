; Keywords
[
  "function"
  "local"
  "module"
  "extern"
  "struct"
  "impl"
  "enum"
  "trait"
  "use"
  "as"
  "is"
  "ref"
  "for"
  "in"
  "while"
  "do"
  "if"
  "then"
  "elseif"
  "else"
  "match"
  "case"
  "return"
  "end"
] @keyword

(visibility) @keyword.storage

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
  "in"
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

; Storage / visibility keywords
[
  "local"
  "ref"
  "module"
  "extern"
] @keyword.storage

; Operators
[
  "+"
  "-"
  "*"
  "/"
  "%"
  "^"
  "=="
  "!="
  "~="
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
  ".."
  "|"
  "&"
  "?"
] @operator

; Special operators
":" @punctuation.delimiter
"." @punctuation.delimiter
"is" @keyword.operator
"as" @keyword.operator

; Delimiters
[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

"," @punctuation.delimiter

; Function declarations
(function_declaration
  name: (identifier) @function)

(function_declaration
  name: (scoped_type_identifier) @function)

(method_identifier
  receiver: (identifier) @type
  method: (identifier) @function.method)

(method_identifier
  receiver: (scoped_type_identifier) @type
  method: (identifier) @function.method)

(trait_method
  name: (identifier) @function)

(extern_function
  name: (identifier) @function)

(extern_function
  name: (scoped_type_identifier) @function)

(extern_const
  name: (identifier) @constant)

(extern_const
  name: (scoped_type_identifier) @constant)

; Function calls
(call_expression
  function: (identifier) @function.call)

; Method calls
(method_call_expression
  method: (identifier) @function.method.call)

; Declarations
(struct_declaration
  name: (identifier) @type)

(struct_field
  name: (identifier) @variable.member)

(enum_declaration
  name: (identifier) @type)

(enum_variant
  name: (identifier) @constructor)

(trait_declaration
  name: (identifier) @type)

(const_declaration
  name: (identifier) @constant)

(static_declaration
  name: (identifier) @constant)

(module_declaration
  name: (identifier) @module)

; Type annotations
(primitive_type) @type.builtin

(generic_type
  name: (identifier) @type)

(generic_type
  name: (scoped_type_identifier) @type)

(scoped_type_identifier) @type

(type_parameter
  name: (identifier) @type.parameter)

(type_parameter
  bound: (identifier) @type)

(type_parameter
  bound: (scoped_type_identifier) @type)

; Struct expressions
(struct_expression
  type: (identifier) @constructor)

(struct_expression
  type: (scoped_type_identifier) @constructor)

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

; Comments
(comment) @comment

; Pattern matching
(wildcard_pattern) @variable.builtin

(enum_pattern
  variant: (identifier) @constructor)

(enum_pattern
  variant: (scoped_type_identifier) @constructor)

(struct_pattern
  type: (identifier) @constructor)

(struct_pattern
  type: (scoped_type_identifier) @constructor)

(struct_pattern_field
  name: (identifier) @variable.member)

; Special identifiers
((identifier) @constant
 (#match? @constant "^[A-Z][A-Z0-9_]*$"))

; Built-in types
[
  "int"
  "float"
  "bool"
  "string"
  "unknown"
] @type.builtin

; Built-in functions
((identifier) @function.builtin
 (#match? @function.builtin "^(print|println|type|tostring)$"))
