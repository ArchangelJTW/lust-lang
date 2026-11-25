; Lust code outline queries for Zed
; Grounded strictly in tree-sitter-lust/src/node-types.json.

(doc_comment) @annotation

(module_declaration
  "module" @context
  name: (_) @name) @item

(struct_declaration
  (visibility)? @context
  "struct" @context
  name: (identifier) @name) @item

(struct_field
  (visibility)? @context
  name: (_) @name) @item

(enum_declaration
  (visibility)? @context
  "enum" @context
  name: (identifier) @name) @item

(enum_variant
  name: (_) @name) @item

(trait_declaration
  (visibility)? @context
  "trait" @context
  name: (identifier) @name) @item

(trait_method
  "function" @context
  name: (_) @name) @item

(impl_block
  "impl" @context
  trait: (_)? @name
  "for"? @context
  type: (_) @name) @item

(function_declaration
  (visibility)? @context
  "function" @context
  name: (_) @name) @item

(extern_declaration
  (visibility)? @context
  "extern" @name
  abi: (string)? @context.extra) @item

(extern_function
  "function" @context
  name: (_) @name) @item

(extern_const
  "const" @context
  name: (_) @name) @item
