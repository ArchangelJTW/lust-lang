; Lust auto-indentation queries for Zed
; Grounded strictly in tree-sitter-lust/src/node-types.json.

; Keyword-delimited blocks
(function_declaration "end" @end) @indent
(lambda_expression "end" @end) @indent
(struct_declaration "end" @end) @indent
(enum_declaration "end" @end) @indent
(trait_declaration "end" @end) @indent
(trait_method "end" @end) @indent
(impl_block "end" @end) @indent
(if_statement "end" @end) @indent
(if_statement
  (elseif_clause "then" @start)
  "end" @end) @indent
(if_statement
  (else_clause "else" @start)
  "end" @end) @indent
(while_statement "end" @end) @indent
(for_numeric_statement "end" @end) @indent
(for_in_statement "end" @end) @indent
(do_statement "end" @end) @indent
(extern_declaration "end" @end) @indent
(extern_declaration "}" @end) @indent

; Outdent branch headers inside `if ... end`
(elseif_clause "elseif" @outdent)
(else_clause "else" @outdent)

; Bracket-delimited lists, literals, and expressions
(parameter_list ")" @end) @indent
(extern_parameter_list ")" @end) @indent
(argument_list ")" @end) @indent
(tuple_expression ")" @end) @indent
(tuple_type ")" @end) @indent
(parenthesized_expression ")" @end) @indent
(array_expression "]" @end) @indent
(struct_expression "}" @end) @indent
(struct_pattern "}" @end) @indent
(map_expression "}" @end) @indent
(use_group "}" @end) @indent
