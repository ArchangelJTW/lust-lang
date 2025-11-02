module.exports = grammar({
  name: 'lust',

  extras: $ => [
    /\s/,
    $.comment,
  ],

  word: $ => $.identifier,

  conflicts: $ => [
    [$.pattern, $.enum_pattern],
    [$.enum_pattern, $.type_annotation],
    [$.enum_pattern, $.type_annotation, $.generic_type],
    [$.type_annotation, $.generic_type],
  ],

  rules: {
    source_file: $ => repeat($._statement),

    _statement: $ => choice(
      $.function_declaration,
      $.struct_declaration,
      $.enum_declaration,
      $.impl_block,
      $.trait_declaration,
      $.use_declaration,
      $.local_declaration,
      $.assignment,
      $.expression_statement,
      $.if_statement,
      $.while_statement,
      $.for_statement,
      $.match_statement,
      $.return_statement,
      $.break_statement,
      $.continue_statement,
    ),

    // Comments
    comment: $ => token(choice(
      seq(
        '--[[',
        repeat(choice(
          /[^\]]/,
          seq(']', /[^\]]/)
        )),
        ']]'
      ),
      seq('--', /[^\n]*/),
      seq('#', /[^\n]*/),
    )),

    // Function declaration
    function_declaration: $ => seq(
      'function',
      field('name', $.identifier),
      optional($.type_parameters),
      field('parameters', $.parameter_list),
      optional(seq(':', field('return_type', $.type_annotation))),
      repeat($._statement),
      'end'
    ),

    parameter_list: $ => seq(
      '(',
      optional(seq(
        $.parameter,
        repeat(seq(',', $.parameter)),
        optional(',')
      )),
      ')'
    ),

    parameter: $ => seq(
      field('name', $.identifier),
      optional(seq(':', field('type', $.type_annotation)))
    ),

    // Type parameters (generics)
    type_parameters: $ => seq(
      '<',
      seq(
        $.identifier,
        repeat(seq(',', $.identifier)),
        optional(',')
      ),
      '>'
    ),

    // Struct declaration
    struct_declaration: $ => seq(
      'struct',
      field('name', $.identifier),
      optional($.type_parameters),
      repeat($.struct_field),
      'end'
    ),

    struct_field: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $.type_annotation)
    ),

    // Enum declaration
    enum_declaration: $ => seq(
      'enum',
      field('name', $.identifier),
      optional($.type_parameters),
      repeat(seq($.enum_variant, optional(','))),
      'end'
    ),

    enum_variant: $ => seq(
      field('name', $.identifier),
      optional(seq(
        '(',
        optional(seq(
          $.type_annotation,
          repeat(seq(',', $.type_annotation)),
          optional(',')
        )),
        ')'
      ))
    ),

    // Impl block
    impl_block: $ => seq(
      'impl',
      optional(seq($.identifier, 'for')),
      field('type', $.identifier),
      repeat($.function_declaration),
      'end'
    ),

    // Trait declaration
    trait_declaration: $ => seq(
      'trait',
      field('name', $.identifier),
      optional($.type_parameters),
      repeat($.trait_method),
      'end'
    ),

    trait_method: $ => seq(
      'function',
      field('name', $.identifier),
      field('parameters', $.parameter_list),
      optional(seq(':', field('return_type', $.type_annotation)))
    ),

    // Use declaration
    use_declaration: $ => seq(
      'use',
      field('kind', optional('type')),
      field('tree', choice(
        $.use_glob,
        $.use_group,
        $.use_path
      ))
    ),

    use_path: $ => seq(
      field('path', $.use_path_segments),
      optional(seq('as', field('alias', $.identifier)))
    ),

    use_glob: $ => seq(
      field('path', $.use_path_segments),
      '.',
      '*'
    ),

    use_group: $ => seq(
      field('path', $.use_path_segments),
      '.',
      '{',
      optional(choice(
        '*',
        seq(
          $.use_group_item,
          repeat(seq(',', $.use_group_item)),
          optional(',')
        )
      )),
      '}'
    ),

    use_group_item: $ => seq(
      field('name', $.identifier),
      optional(seq('as', field('alias', $.identifier)))
    ),

    use_path_segments: $ => prec.left(seq(
      $.identifier,
      repeat(seq('.', $.identifier))
    )),

    // Local variable declaration
    local_declaration: $ => seq(
      'local',
      optional('mut'),
      field('bindings', $.binding_list),
      optional(seq('=', field('values', $.expression_list)))
    ),

    binding_list: $ => seq(
      $.binding,
      repeat(seq(',', $.binding))
    ),

    binding: $ => seq(
      field('name', $.identifier),
      optional(seq(':', field('type', $.type_annotation)))
    ),

    expression_list: $ => seq(
      $._expression,
      repeat(seq(',', $._expression))
    ),

    assignment_target: $ => choice(
      $.identifier,
      $.field_access,
      $.index_access
    ),

    assignment_targets: $ => seq(
      field('target', $.assignment_target),
      repeat(seq(',', field('target', $.assignment_target)))
    ),

    // Assignment
    assignment: $ => choice(
      seq(
        field('targets', $.assignment_targets),
        '=',
        field('values', $.expression_list)
      ),
      seq(
        field('target', $.assignment_target),
        choice('+=', '-=', '*=', '/=', '%='),
        field('value', $._expression)
      )
    ),

    // Expression statement
    expression_statement: $ => $._expression,

    // If statement
    if_statement: $ => seq(
      'if',
      field('condition', $._expression),
      'then',
      repeat($._statement),
      repeat($.elseif_clause),
      optional($.else_clause),
      'end'
    ),

    elseif_clause: $ => seq(
      'elseif',
      field('condition', $._expression),
      'then',
      repeat($._statement)
    ),

    else_clause: $ => seq(
      'else',
      repeat($._statement)
    ),

    // While statement
    while_statement: $ => seq(
      'while',
      field('condition', $._expression),
      'do',
      repeat($._statement),
      'end'
    ),

    // For statement
    for_statement: $ => seq(
      'for',
      field('variable', $.identifier),
      '=',
      field('start', $._expression),
      ',',
      field('end', $._expression),
      optional(seq(',', field('step', $._expression))),
      'do',
      repeat($._statement),
      'end'
    ),

    // Match statement
    match_statement: $ => seq(
      'match',
      field('value', $._expression),
      'do',
      repeat($.match_case),
      'end'
    ),

    match_case: $ => seq(
      'case',
      field('pattern', $.pattern),
      'then',
      repeat($._statement)
    ),

    pattern: $ => choice(
      $.identifier,
      $.enum_pattern,
      $.literal,
    ),

    enum_pattern: $ => seq(
      field('variant', $.identifier),
      optional(seq(
        '(',
        optional(seq(
          $.identifier,
          repeat(seq(',', $.identifier)),
          optional(',')
        )),
        ')'
      ))
    ),

    is_expression: $ => prec.left(9, seq(
      field('value', $._expression),
      'is',
      field('pattern', choice(
        $.enum_pattern,
        $.type_annotation
      ))
    )),

    // Control flow
    return_statement: $ => prec.right(1, seq(
      'return',
      optional($._expression)
    )),

    break_statement: $ => 'break',

    continue_statement: $ => 'continue',

    // Expressions
    _expression: $ => choice(
      $.literal,
      $.identifier,
      $.is_expression,
      $.binary_expression,
      $.unary_expression,
      $.call_expression,
      $.method_call_expression,
      $.field_access,
      $.index_access,
      $.struct_expression,
      $.array_expression,
      $.map_expression,
      $.lambda_expression,
      $.parenthesized_expression,
    ),

    // Binary expressions
    binary_expression: $ => choice(
      ...[
        ['or', 1],
        ['and', 2],
        ['==', 3],
        ['!=', 3],
        ['<', 4],
        ['<=', 4],
        ['>', 4],
        ['>=', 4],
        ['..', 5],
        ['+', 6],
        ['-', 6],
        ['*', 7],
        ['/', 7],
        ['%', 7],
      ].map(([operator, precedence]) =>
        prec.left(precedence, seq(
          field('left', $._expression),
          field('operator', operator),
          field('right', $._expression)
        ))
      )
    ),

    // Unary expressions
    unary_expression: $ => prec(8, seq(
      field('operator', choice('not', '-', '#')),
      field('operand', $._expression)
    )),

    // Function call
    call_expression: $ => prec(10, seq(
      field('function', $._expression),
      field('arguments', $.argument_list)
    )),

    argument_list: $ => seq(
      '(',
      optional(seq(
        $._expression,
        repeat(seq(',', $._expression)),
        optional(',')
      )),
      ')'
    ),

    // Method call
    method_call_expression: $ => prec(10, seq(
      field('object', $._expression),
      ':',
      field('method', $.identifier),
      optional($.type_arguments),
      field('arguments', $.argument_list)
    )),

    type_arguments: $ => seq(
      '<',
      seq(
        $.type_annotation,
        repeat(seq(',', $.type_annotation)),
        optional(',')
      ),
      '>'
    ),

    // Field access
    field_access: $ => prec(10, seq(
      field('object', $._expression),
      '.',
      field('field', $.identifier)
    )),

    // Index access
    index_access: $ => prec(10, seq(
      field('object', $._expression),
      '[',
      field('index', $._expression),
      ']'
    )),

    // Struct expression
    struct_expression: $ => prec(2, seq(
      field('type', $.identifier),
      '{',
      optional(seq(
        $.struct_field_init,
        repeat(seq(',', $.struct_field_init)),
        optional(',')
      )),
      '}'
    )),

    struct_field_init: $ => seq(
      field('name', $.identifier),
      '=',
      field('value', $._expression)
    ),

    // Array expression
    array_expression: $ => seq(
      '[',
      optional(seq(
        $._expression,
        repeat(seq(',', $._expression)),
        optional(',')
      )),
      ']'
    ),

    // Map expression
    map_expression: $ => seq(
      '{',
      optional(seq(
        $.map_entry,
        repeat(seq(',', $.map_entry)),
        optional(',')
      )),
      '}'
    ),

    map_entry: $ => choice(
      seq(
        '[',
        field('key', $._expression),
        ']',
        '=',
        field('value', $._expression)
      ),
      seq(
        field('key', $.identifier),
        '=',
        field('value', $._expression)
      )
    ),

    // Lambda expression
    lambda_expression: $ => seq(
      'function',
      field('parameters', $.parameter_list),
      optional(seq(':', field('return_type', $.type_annotation))),
      repeat($._statement),
      'end'
    ),

    // Parenthesized expression
    parenthesized_expression: $ => seq(
      '(',
      $._expression,
      ')'
    ),

    // Type annotations
    type_annotation: $ => choice(
      $.primitive_type,
      $.generic_type,
      $.function_type,
      $.identifier,
    ),

    primitive_type: $ => choice(
      'int',
      'float',
      'bool',
      'string',
      'unknown',
      prec(1, 'nil'),
    ),

    generic_type: $ => seq(
      field('name', $.identifier),
      '<',
      seq(
        $.type_annotation,
        repeat(seq(',', $.type_annotation)),
        optional(',')
      ),
      '>'
    ),

    function_type: $ => seq(
      'fn',
      '(',
      optional(seq(
        $.type_annotation,
        repeat(seq(',', $.type_annotation)),
        optional(',')
      )),
      ')',
      '->',
      $.type_annotation
    ),

    // Literals
    literal: $ => choice(
      $.number,
      $.string,
      $.boolean,
      $.nil,
    ),

    number: $ => token(choice(
      /\d+/,
      /\d+\.\d+/,
    )),

    string: $ => token(seq(
      '"',
      repeat(choice(
        /[^"\\]/,
        /\\./,
      )),
      '"'
    )),

    boolean: $ => choice('true', 'false'),

    nil: $ => 'nil',

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,
  }
});
