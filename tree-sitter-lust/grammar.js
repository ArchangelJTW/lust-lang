module.exports = grammar({
  name: 'lust',

  extras: $ => [
    /\s/,
    $.comment,
  ],

  word: $ => $.identifier,

  conflicts: $ => [
    [$.pattern, $.enum_pattern],
    [$.pattern, $.enum_pattern, $._primary_type],
    [$.pattern, $.enum_pattern, $.struct_pattern, $._primary_type],
    [$.pattern, $.enum_pattern, $._primary_type, $.generic_type],
    [$.scoped_type_identifier, $.pattern, $.enum_pattern, $._primary_type],
    [$.enum_pattern, $._primary_type],
    [$.enum_pattern, $.struct_pattern, $._primary_type],
    [$.enum_pattern, $._primary_type, $.generic_type],
    [$._primary_type, $.generic_type],
    [$._primary_type, $.scoped_type_identifier],
    [$.binary_expression, $.unary_expression, $.call_expression],
    [$.binary_expression, $.call_expression],
    [$._expression, $.generic_type],
    [$._expression, $._primary_type],
    [$.binding, $.assignment_target],
    [$.parameter, $._primary_type],
    [$.parameter_list, $.function_type],
    [$.binding, $._expression],
    [$.function_type],
    [$.trait_method],
    [$.wildcard_pattern, $.wildcard_type],
    [$._expression, $.scoped_type_identifier],
  ],

  rules: {
    source_file: $ => repeat($._statement),

    _statement: $ => choice(
      $.function_declaration,
      $.struct_declaration,
      $.enum_declaration,
      $.impl_block,
      $.trait_declaration,
      $.module_declaration,
      $.extern_declaration,
      $.use_declaration,
      $.local_declaration,
      $.assignment,
      $.expression_statement,
      $.if_statement,
      $.while_statement,
      $.for_statement,
      $.do_statement,
      $.match_statement,
      $.return_statement,
      $.break_statement,
      $.continue_statement,
    ),

    visibility: $ => 'pub',

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
      optional($.visibility),
      'function',
      field('name', choice(
        $.identifier,
        $.scoped_type_identifier,
        $.method_identifier,
        $.primitive_type,
      )),
      optional($.type_parameters),
      field('parameters', $.parameter_list),
      optional(seq(':', field('return_type', $.type_annotation))),
      repeat($._statement),
      'end'
    ),

    scoped_type_identifier: $ => prec.left(seq(
      choice($.identifier, $.primitive_type),
      repeat1(seq('.', choice($.identifier, $.primitive_type)))
    )),

    method_identifier: $ => seq(
      field('receiver', choice($.identifier, $.scoped_type_identifier, $.primitive_type)),
      ':',
      field('method', choice($.identifier, $.primitive_type))
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
      field('name', choice($.identifier, $.primitive_type)),
      optional(seq(':', field('type', $.type_annotation)))
    ),

    // Type parameters (generics)
    type_parameters: $ => seq(
      '<',
      seq(
        $.type_parameter,
        repeat(seq(',', $.type_parameter)),
        optional(',')
      ),
      '>'
    ),

    type_parameter: $ => seq(
      field('name', $.identifier),
      optional(seq(
        ':',
        field('bound', choice($.identifier, $.scoped_type_identifier, $.primitive_type)),
        repeat(seq('+', field('bound', choice($.identifier, $.scoped_type_identifier, $.primitive_type))))
      ))
    ),

    module_declaration: $ => seq(
      'module',
      field('name', choice($.identifier, $.scoped_type_identifier, $.primitive_type))
    ),

    // Struct declaration
    struct_declaration: $ => seq(
      optional($.visibility),
      'struct',
      field('name', $.identifier),
      optional($.type_parameters),
      repeat($.struct_field),
      'end'
    ),

    struct_field: $ => seq(
      optional($.visibility),
      field('name', choice($.identifier, $.primitive_type)),
      ':',
      optional('ref'),
      field('type', $.type_annotation),
      optional(',')
    ),

    // Enum declaration
    enum_declaration: $ => seq(
      optional($.visibility),
      'enum',
      field('name', $.identifier),
      optional($.type_parameters),
      repeat($.enum_variant),
      'end'
    ),

    enum_variant: $ => seq(
      field('name', choice($.identifier, $.primitive_type)),
      optional(seq(
        '(',
        optional(seq(
          $.type_annotation,
          repeat(seq(',', $.type_annotation)),
          optional(',')
        )),
        ')'
      )),
      optional(',')
    ),

    // Impl block
    impl_block: $ => seq(
      'impl',
      optional($.type_parameters),
      choice(
        seq(field('trait', $.type_annotation), 'for', field('type', $.type_annotation)),
        field('type', $.type_annotation)
      ),
      repeat($.function_declaration),
      'end'
    ),

    // Trait declaration
    trait_declaration: $ => seq(
      optional($.visibility),
      'trait',
      field('name', $.identifier),
      optional($.type_parameters),
      repeat($.trait_method),
      'end'
    ),

    trait_method: $ => seq(
      'function',
      field('name', choice($.identifier, $.primitive_type)),
      optional($.type_parameters),
      field('parameters', $.parameter_list),
      optional(seq(':', field('return_type', $.type_annotation))),
      optional(seq(
        repeat($._statement),
        'end'
      ))
    ),

    // Extern declaration
    extern_declaration: $ => seq(
      optional($.visibility),
      'extern',
      optional(field('abi', $.string)),
      choice(
        seq('{', repeat($._extern_item), '}'),
        seq(repeat($._extern_item), 'end')
      )
    ),

    _extern_item: $ => choice(
      $.extern_function,
      $.extern_const,
      $.struct_declaration,
      $.enum_declaration,
    ),

    extern_function: $ => seq(
      'function',
      field('name', choice(
        $.identifier,
        $.scoped_type_identifier,
        $.method_identifier,
        $.primitive_type,
      )),
      field('parameters', $.extern_parameter_list),
      optional(seq(':', field('return_type', $.type_annotation)))
    ),

    extern_parameter_list: $ => seq(
      '(',
      optional(seq(
        $.type_annotation,
        repeat(seq(',', $.type_annotation)),
        optional(',')
      )),
      ')'
    ),

    extern_const: $ => seq(
      'const',
      field('name', choice(
        $.identifier,
        $.scoped_type_identifier,
        $.primitive_type,
      )),
      ':',
      field('type', $.type_annotation)
    ),

    // Use declaration
    use_declaration: $ => seq(
      optional($.visibility),
      'use',
      choice(
        $.use_glob,
        $.use_group,
        $.use_path,
      )
    ),

    use_path: $ => seq(
      field('path', choice($.identifier, $.primitive_type)),
      repeat(seq('.', field('path', choice($.identifier, $.primitive_type)))),
      optional(seq('as', field('alias', choice($.identifier, $.primitive_type))))
    ),

    use_glob: $ => seq(
      field('path', choice($.identifier, $.primitive_type)),
      repeat(seq('.', field('path', choice($.identifier, $.primitive_type)))),
      '.',
      '*'
    ),

    use_group: $ => seq(
      field('path', choice($.identifier, $.primitive_type)),
      repeat(seq('.', field('path', choice($.identifier, $.primitive_type)))),
      '.',
      '{',
      optional(seq(
        $.use_group_item,
        repeat(seq(',', $.use_group_item)),
        optional(',')
      )),
      '}'
    ),

    use_group_item: $ => choice(
      '*',
      seq(
        field('path', choice($.identifier, $.primitive_type)),
        repeat(seq('.', field('path', choice($.identifier, $.primitive_type)))),
        optional(seq('as', field('alias', choice($.identifier, $.primitive_type))))
      )
    ),

    // Local variable declaration
    local_declaration: $ => seq(
      'local',
      field('bindings', $.binding_list),
      optional(seq('=', field('values', $.expression_list)))
    ),

    binding_list: $ => seq(
      $.binding,
      repeat(seq(',', $.binding))
    ),

    binding: $ => seq(
      field('name', choice($.identifier, $.primitive_type)),
      optional(seq(':', field('type', $.type_annotation)))
    ),

    expression_list: $ => seq(
      $._expression,
      repeat(seq(',', $._expression))
    ),

    assignment_target: $ => choice(
      $.identifier,
      $.primitive_type,
      $.field_access,
      $.index_access
    ),

    assignment_targets: $ => seq(
      field('target', $.assignment_target),
      repeat(seq(',', field('target', $.assignment_target)))
    ),

    // Assignment & implicit local
    assignment: $ => choice(
      seq(
        field('targets', $.assignment_targets),
        '=',
        field('values', $.expression_list)
      ),
      seq(
        field('target', $.assignment_target),
        choice('+=', '-=', '*=', '/='),
        field('value', $._expression)
      ),
      seq(
        field('bindings', $.binding_list),
        '=',
        field('values', $.expression_list)
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
    for_statement: $ => choice(
      $.for_numeric_statement,
      $.for_in_statement
    ),

    for_numeric_statement: $ => seq(
      'for',
      field('variable', choice($.identifier, $.primitive_type)),
      '=',
      field('start', $._expression),
      ',',
      field('end', $._expression),
      optional(seq(',', field('step', $._expression))),
      'do',
      repeat($._statement),
      'end'
    ),

    for_in_statement: $ => seq(
      'for',
      field('variables', seq(
        choice($.identifier, $.primitive_type),
        repeat(seq(',', choice($.identifier, $.primitive_type)))
      )),
      'in',
      field('iterator', $._expression),
      'do',
      repeat($._statement),
      'end'
    ),

    // Do block statement
    do_statement: $ => seq(
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
      $.wildcard_pattern,
      $.enum_pattern,
      $.struct_pattern,
      $.type_check_pattern,
      $.literal,
      $.identifier,
      $.primitive_type,
    ),

    wildcard_pattern: $ => '_',

    enum_pattern: $ => seq(
      field('variant', choice($.identifier, $.scoped_type_identifier, $.primitive_type)),
      optional(seq(
        '(',
        optional(seq(
          $.pattern,
          repeat(seq(',', $.pattern)),
          optional(',')
        )),
        ')'
      ))
    ),

    struct_pattern: $ => seq(
      field('type', choice($.identifier, $.scoped_type_identifier, $.primitive_type)),
      '{',
      optional(seq(
        $.struct_pattern_field,
        repeat(seq(',', $.struct_pattern_field)),
        optional(',')
      )),
      '}'
    ),

    struct_pattern_field: $ => choice(
      field('name', choice($.identifier, $.primitive_type)),
      seq(field('name', choice($.identifier, $.primitive_type)), '=', field('pattern', $.pattern))
    ),

    type_check_pattern: $ => seq(
      'as',
      field('type', $.type_annotation)
    ),

    is_expression: $ => prec.left(10, seq(
      field('value', $._expression),
      'is',
      field('pattern', choice(
        $.pattern,
        $.type_annotation
      ))
    )),

    cast_expression: $ => prec.left(10, seq(
      field('value', $._expression),
      'as',
      field('type', $.type_annotation)
    )),

    try_expression: $ => prec.left(10, seq(
      field('value', $._expression),
      '?'
    )),

    // Control flow
    return_statement: $ => prec.right(1, seq(
      'return',
      optional($.expression_list)
    )),

    break_statement: $ => 'break',

    continue_statement: $ => 'continue',

    // Expressions
    _expression: $ => choice(
      $.literal,
      $.identifier,
      $.primitive_type,
      $.cast_expression,
      $.is_expression,
      $.try_expression,
      $.binary_expression,
      $.unary_expression,
      $.call_expression,
      $.method_call_expression,
      $.field_access,
      $.index_access,
      $.struct_expression,
      $.array_expression,
      $.map_expression,
      $.tuple_expression,
      $.lambda_expression,
      $.parenthesized_expression,
    ),

    // Binary expressions
    binary_expression: $ => choice(
      prec.left(1, seq(field('left', $._expression), field('operator', 'or'), field('right', $._expression))),
      prec.left(2, seq(field('left', $._expression), field('operator', 'and'), field('right', $._expression))),
      prec.left(3, seq(field('left', $._expression), field('operator', choice('==', '!=', '~=')), field('right', $._expression))),
      prec.left(4, seq(field('left', $._expression), field('operator', choice('<', '<=', '>', '>=')), field('right', $._expression))),
      prec.left(5, seq(field('left', $._expression), field('operator', '..'), field('right', $._expression))),
      prec.left(6, seq(field('left', $._expression), field('operator', choice('+', '-')), field('right', $._expression))),
      prec.left(7, seq(field('left', $._expression), field('operator', choice('*', '/', '%')), field('right', $._expression))),
      prec.right(8, seq(field('left', $._expression), field('operator', '^'), field('right', $._expression))),
    ),

    // Unary expressions
    unary_expression: $ => prec(9, seq(
      field('operator', choice('not', '-')),
      field('operand', $._expression)
    )),

    // Function call
    call_expression: $ => prec(10, seq(
      field('function', $._expression),
      optional($.type_arguments),
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
      field('method', choice($.identifier, $.primitive_type)),
      optional($.type_arguments),
      field('arguments', $.argument_list)
    )),

    type_arguments: $ => prec.dynamic(1, seq(
      '<',
      seq(
        $.type_annotation,
        repeat(seq(',', $.type_annotation)),
        optional(',')
      ),
      '>'
    )),

    // Field access
    field_access: $ => prec(10, seq(
      field('object', $._expression),
      '.',
      field('field', choice($.identifier, $.primitive_type))
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
      field('type', choice($.identifier, $.scoped_type_identifier, $.primitive_type)),
      '{',
      optional(seq(
        $.struct_field_init,
        repeat(seq(',', $.struct_field_init)),
        optional(',')
      )),
      '}'
    )),

    struct_field_init: $ => seq(
      field('name', choice($.identifier, $.primitive_type)),
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
        field('key', choice($.identifier, $.primitive_type)),
        '=',
        field('value', $._expression)
      )
    ),

    tuple_expression: $ => seq(
      '(',
      $._expression,
      repeat1(seq(',', $._expression)),
      optional(','),
      ')'
    ),

    // Lambda expression
    lambda_expression: $ => seq(
      'function',
      field('parameters', $.parameter_list),
      optional(choice(
        seq(':', field('return_type', $.type_annotation)),
        seq('->', field('return_type', $.type_annotation))
      )),
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
      $.union_type,
      $._primary_type,
    ),

    union_type: $ => prec.left(1, seq(
      $.type_annotation,
      '|',
      $.type_annotation
    )),

    _primary_type: $ => choice(
      $.primitive_type,
      $.unit_type,
      $.tuple_type,
      $.ref_type,
      $.pointer_type,
      $.function_type,
      $.generic_type,
      $.scoped_type_identifier,
      $.identifier,
      $.wildcard_type,
    ),

    primitive_type: $ => choice(
      'int',
      'float',
      'bool',
      'string',
      'unknown',
    ),

    unit_type: $ => seq('(', ')'),

    tuple_type: $ => seq(
      '(',
      $.type_annotation,
      repeat1(seq(',', $.type_annotation)),
      optional(','),
      ')'
    ),

    ref_type: $ => seq(
      '&',
      field('type', $.type_annotation)
    ),

    pointer_type: $ => seq(
      '*',
      field('type', $.type_annotation)
    ),

    generic_type: $ => seq(
      field('name', choice($.identifier, $.scoped_type_identifier, $.primitive_type)),
      '<',
      seq(
        $.type_annotation,
        repeat(seq(',', $.type_annotation)),
        optional(',')
      ),
      '>'
    ),

    function_type: $ => seq(
      'function',
      '(',
      optional(seq(
        $.type_annotation,
        repeat(seq(',', $.type_annotation)),
        optional(',')
      )),
      ')',
      optional(seq(':', field('return_type', $.type_annotation)))
    ),

    wildcard_type: $ => '_',

    // Literals
    literal: $ => choice(
      $.number,
      $.string,
      $.boolean,
    ),

    number: $ => token(choice(
      /\d+(\.\d+)?([eE][+-]?\d+)?/,
      /0[xX][0-9a-fA-F]+/,
      /0[bB][01]+/,
      /0[oO][0-7]+/,
    )),

    string: $ => token(choice(
      seq(
        '"',
        repeat(choice(
          /[^"\\]/,
          /\\./,
        )),
        '"'
      ),
      seq(
        "'",
        repeat(choice(
          /[^'\\]/,
          /\\./,
        )),
        "'"
      )
    )),

    boolean: $ => choice('true', 'false'),

    identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,
  }
});
