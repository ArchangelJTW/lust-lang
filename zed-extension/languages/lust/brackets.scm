; Lust bracket matching queries for Zed
; Grounded strictly in tree-sitter-lust/src/node-types.json.

("(" @open ")" @close)
("[" @open "]" @close)
("{" @open "}" @close)

(type_parameters "<" @open ">" @close)
(type_arguments "<" @open ">" @close)
(generic_type "<" @open ">" @close)

; Block keyword pairs (excluded from rainbow bracket colorization)
((function_declaration "function" @open "end" @close) (#set! rainbow.exclude))
((lambda_expression "function" @open "end" @close) (#set! rainbow.exclude))
((struct_declaration "struct" @open "end" @close) (#set! rainbow.exclude))
((enum_declaration "enum" @open "end" @close) (#set! rainbow.exclude))
((trait_declaration "trait" @open "end" @close) (#set! rainbow.exclude))
((trait_method "function" @open "end" @close) (#set! rainbow.exclude))
((impl_block "impl" @open "end" @close) (#set! rainbow.exclude))
((if_statement "then" @open "end" @close) (#set! rainbow.exclude))
((while_statement "do" @open "end" @close) (#set! rainbow.exclude))
((for_numeric_statement "do" @open "end" @close) (#set! rainbow.exclude))
((for_in_statement "do" @open "end" @close) (#set! rainbow.exclude))
((do_statement "do" @open "end" @close) (#set! rainbow.exclude))
((extern_declaration "extern" @open "end" @close) (#set! rainbow.exclude))
