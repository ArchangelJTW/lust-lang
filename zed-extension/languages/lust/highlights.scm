; Lust syntax highlighting queries for Zed
; Grounded strictly in tree-sitter-lust/src/node-types.json.
; Ordered general-to-specific (later rules override earlier captures in Zed).

; ---------------------------------------------------------------------------
; Identifiers & conventions (general fallback first)
; ---------------------------------------------------------------------------

(identifier) @variable

(binding
  name: (identifier) @variable)

(for_numeric_statement
  variable: (identifier) @variable)

(for_in_statement
  variables: (identifier) @variable)

(parameter
  name: (identifier) @variable.parameter)

(extern_parameter
  name: (identifier) @variable.parameter)

; PascalCase types and ALL_CAPS constants (min 2 chars so single-letter T/K/V stay @type)
((identifier) @type
  (#match? @type "^[A-Z][a-zA-Z0-9_]*$"))

((identifier) @constant
  (#match? @constant "^[A-Z][A-Z0-9_]+$"))

; ---------------------------------------------------------------------------
; Types & declarations
; ---------------------------------------------------------------------------

(primitive_type) @type.builtin

(wildcard_type) @type.builtin

; Scoped type identifiers: prefix segments are @namespace, final segment is @type
(scoped_type_identifier
  (identifier) @namespace)

(scoped_type_identifier
  (identifier) @type .)

(type_annotation
  (identifier) @type)

(generic_type
  name: (identifier) @type)

(type_parameter
  name: (identifier) @type)

(type_parameter
  bound: (identifier) @type)

(struct_declaration
  name: (identifier) @type)

(enum_declaration
  name: (identifier) @type)

(trait_declaration
  name: (identifier) @type)

; Built-in generic/collection types (placed after @type rules so Option<T>,
; Result<T, E>, Array, Map<K, V> resolve to @type.builtin in type positions)
((identifier) @type.builtin
  (#match? @type.builtin "^(Option|Result|Array|Map)$"))

; ---------------------------------------------------------------------------
; Modules & imports
; ---------------------------------------------------------------------------

(module_declaration
  name: (identifier) @namespace)

(module_declaration
  name: (scoped_type_identifier
    (identifier) @namespace))

(use_path
  path: (identifier) @namespace)

(use_path
  alias: (identifier) @type)

(use_glob
  path: (identifier) @namespace)

(use_group
  path: (identifier) @namespace)

(use_group_item
  path: (identifier) @namespace)

(use_group_item
  alias: (identifier) @type)

; ---------------------------------------------------------------------------
; Properties, fields & keys
; ---------------------------------------------------------------------------

(struct_field
  name: (identifier) @property)

(struct_field_init
  name: (identifier) @property)

(struct_pattern_field
  name: (identifier) @property)

(field_access
  field: (identifier) @property)

; Bare identifier map keys `{ foo = 1 }` are @property; computed keys `{ [k] = 1 }` keep expression captures
(map_entry
  key: (identifier) @property)

(map_entry
  "["
  key: (identifier) @variable
  "]")

(map_entry
  "["
  key: (identifier) @type
  "]"
  (#match? @type "^[A-Z][a-zA-Z0-9_]*$"))

(map_entry
  "["
  key: (identifier) @constant
  "]"
  (#match? @constant "^[A-Z][A-Z0-9_]+$"))

; ---------------------------------------------------------------------------
; Constructors & constants
; ---------------------------------------------------------------------------

(enum_variant
  name: (identifier) @constructor)

(enum_pattern
  variant: (identifier) @constructor)

(enum_pattern
  variant: (scoped_type_identifier
    (identifier) @type
    (#match? @type "^[A-Z]")))

(enum_pattern
  variant: (scoped_type_identifier
    (identifier) @constructor .))

(struct_expression
  type: (identifier) @constructor)

(struct_expression
  type: (scoped_type_identifier
    (identifier) @constructor .))

(struct_pattern
  type: (identifier) @constructor)

(struct_pattern
  type: (scoped_type_identifier
    (identifier) @constructor .))

(extern_const
  name: (identifier) @constant)

(extern_const
  name: (scoped_type_identifier
    (identifier) @constant .))

; ---------------------------------------------------------------------------
; Functions & methods
; ---------------------------------------------------------------------------

(function_declaration
  name: (identifier) @function)

(function_declaration
  name: (scoped_type_identifier
    (identifier) @type
    (#match? @type "^[A-Z]")))

(function_declaration
  name: (scoped_type_identifier
    (identifier) @function .))

(extern_function
  name: (identifier) @function)

(extern_function
  name: (scoped_type_identifier
    (identifier) @type
    (#match? @type "^[A-Z]")))

(extern_function
  name: (scoped_type_identifier
    (identifier) @function .))

(impl_block
  (function_declaration
    name: (identifier) @function.method))

(method_identifier
  receiver: (identifier) @type)

(method_identifier
  method: (identifier) @function.method)

(trait_method
  name: (identifier) @function.method)

(call_expression
  function: (identifier) @function)

(call_expression
  function: (field_access
    field: (identifier) @function))

(method_call_expression
  method: (identifier) @function.method)

; Built-in functions (matched in call position to avoid shadowing local variables)
(call_expression
  function: (identifier) @function.builtin
  (#match? @function.builtin "^(print|println|type|tostring|assert|error|pcall|xpcall|pairs|ipairs|len|panic)$"))

; ---------------------------------------------------------------------------
; Special variables, built-in constants & reserved modern lexer keywords
; ---------------------------------------------------------------------------

((identifier) @variable.special
  (#eq? @variable.special "self"))

((identifier) @constant.builtin
  (#eq? @constant.builtin "nil"))

(wildcard_pattern) @variable.special

; Reserved lexer keywords in lust-rs not yet modeled as grammar rules
((identifier) @keyword
  (#match? @keyword "^(defer|errdefer|scoped)$"))

; ---------------------------------------------------------------------------
; Keywords
; ---------------------------------------------------------------------------

[
  "function"
  "local"
  "module"
  "extern"
  "const"
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
  "return"
  "end"
] @keyword

(visibility) @keyword
(break_statement) @keyword
(continue_statement) @keyword

; ---------------------------------------------------------------------------
; Operators
; ---------------------------------------------------------------------------

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
  "->"
] @operator

; ---------------------------------------------------------------------------
; Punctuation
; ---------------------------------------------------------------------------

[
  ","
  "."
  ":"
] @punctuation.delimiter

[
  "("
  ")"
  "["
  "]"
  "{"
  "}"
] @punctuation.bracket

(type_parameters
  "<" @punctuation.bracket
  ">" @punctuation.bracket)

(type_arguments
  "<" @punctuation.bracket
  ">" @punctuation.bracket)

(generic_type
  "<" @punctuation.bracket
  ">" @punctuation.bracket)

; ---------------------------------------------------------------------------
; Literals & comments
; ---------------------------------------------------------------------------

(number) @number
(string) @string
(boolean) @boolean

(comment) @comment
(doc_comment) @comment.doc
