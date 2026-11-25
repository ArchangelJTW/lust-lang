#!/usr/bin/env node
// Automated verification oracle for vsc-extension TextMate grammar & language configuration.
// Zero external dependencies; works with Node.js >= 18 and Bun.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const grammarPath = path.join(__dirname, 'syntaxes', 'lust.tmLanguage.json');
const langConfigPath = path.join(__dirname, 'language-configuration.json');
const pkgPath = path.join(__dirname, 'package.json');

const grammar = JSON.parse(fs.readFileSync(grammarPath, 'utf8'));
const langConfig = JSON.parse(fs.readFileSync(langConfigPath, 'utf8'));
const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));

// ---------------------------------------------------------------------------
// 1. Metadata & structural validation
// ---------------------------------------------------------------------------
assert.equal(grammar.name, 'Lust');
assert.equal(grammar.scopeName, 'source.lust');
assert.ok(Array.isArray(grammar.patterns) && grammar.patterns.length > 0);
assert.ok(grammar.repository && typeof grammar.repository === 'object');

assert.equal(pkg.name, 'lust-analyzer');
assert.equal(pkg.contributes.languages[0].id, 'lust');
assert.equal(pkg.contributes.grammars[0].scopeName, 'source.lust');
assert.equal(pkg.contributes.grammars[0].path, './syntaxes/lust.tmLanguage.json');

// ---------------------------------------------------------------------------
// 2. Validate every regex compiles cleanly & every #include resolves
// ---------------------------------------------------------------------------
let regexCount = 0;

function validateRegex(pattern, label) {
  assert.equal(typeof pattern, 'string', `${label} must be a string`);
  // Check for Oniguruma-only features that break JS or vice versa
  try {
    new RegExp(pattern, 'g');
    regexCount += 1;
  } catch (err) {
    assert.fail(`Invalid regex in ${label}: ${pattern}\n${err.message}`);
  }
}

function validatePatternList(patterns, contextLabel) {
  assert.ok(Array.isArray(patterns), `${contextLabel} must be an array`);
  for (let i = 0; i < patterns.length; i += 1) {
    const rule = patterns[i];
    const label = `${contextLabel}[${i}]`;
    if (rule.include) {
      assert.ok(rule.include.startsWith('#'), `${label}.include must reference local repository`);
      const refName = rule.include.slice(1);
      assert.ok(
        Object.prototype.hasOwnProperty.call(grammar.repository, refName),
        `${label}.include references missing repository key "${refName}"`,
      );
    }
    if (rule.match !== undefined) {
      validateRegex(rule.match, `${label}.match`);
    }
    if (rule.begin !== undefined) {
      validateRegex(rule.begin, `${label}.begin`);
      assert.ok(rule.end !== undefined, `${label} has begin without end`);
      validateRegex(rule.end, `${label}.end`);
    }
    if (rule.patterns !== undefined) {
      validatePatternList(rule.patterns, `${label}.patterns`);
    }
  }
}

validatePatternList(grammar.patterns, 'grammar.patterns');
for (const [key, entry] of Object.entries(grammar.repository)) {
  validatePatternList(entry.patterns, `repository.${key}.patterns`);
}

validateRegex(langConfig.folding.markers.start, 'language-configuration.folding.markers.start');
validateRegex(langConfig.folding.markers.end, 'language-configuration.folding.markers.end');
validateRegex(
  langConfig.indentationRules.increaseIndentPattern,
  'language-configuration.indentationRules.increaseIndentPattern',
);
validateRegex(
  langConfig.indentationRules.decreaseIndentPattern,
  'language-configuration.indentationRules.decreaseIndentPattern',
);
validateRegex(langConfig.wordPattern, 'language-configuration.wordPattern');

// ---------------------------------------------------------------------------
// 3. Faithful TextMate tokenizer engine (single-line & multi-line begin/end)
// ---------------------------------------------------------------------------
function expandPatterns(patterns) {
  const out = [];
  for (const rule of patterns) {
    if (rule.include) {
      const target = grammar.repository[rule.include.slice(1)];
      out.push(...expandPatterns(target.patterns));
    } else {
      out.push(rule);
    }
  }
  return out;
}

function execWithIndices(patternStr, line, fromIndex) {
  const re = new RegExp(patternStr, 'gd');
  re.lastIndex = fromIndex;
  return re.exec(line);
}

function emitCaptures(tokens, match, captures, baseScopes) {
  if (!captures) {
    return false;
  }
  const indices = match.indices;
  const spans = [];
  for (const [groupKey, spec] of Object.entries(captures)) {
    const idx = Number(groupKey);
    if (!indices[idx]) continue;
    const [start, end] = indices[idx];
    if (start === end) continue;
    spans.push({ start, end, scope: spec.name });
  }
  spans.sort((a, b) => a.start - b.start || a.end - b.end);
  for (const span of spans) {
    tokens.push({
      start: span.start,
      end: span.end,
      text: match.input.slice(span.start, span.end),
      scopes: [...baseScopes, span.scope],
    });
  }
  return spans.length > 0;
}

function tokenizeSource(source) {
  const rootRules = expandPatterns(grammar.patterns);
  const lines = source.split('\n');
  const allLineTokens = [];
  // Stack of active begin/end states: { rule, scopes }
  const stateStack = [{ rule: null, rules: rootRules, scopes: [grammar.scopeName] }];

  for (const line of lines) {
    const tokens = [];
    let pos = 0;

    while (pos <= line.length) {
      const top = stateStack[stateStack.length - 1];
      let bestMatch = null;
      let bestKind = null; // 'end' | 'match' | 'begin'
      let bestRule = null;

      // Check end of current begin/end block first (TextMate semantics: end wins at same index if no earlier rule matches)
      if (top.rule && top.rule.end) {
        const endMatch = execWithIndices(top.rule.end, line, pos);
        if (endMatch) {
          bestMatch = endMatch;
          bestKind = 'end';
          bestRule = top.rule;
        }
      }

      for (const rule of top.rules) {
        if (rule.match !== undefined) {
          const m = execWithIndices(rule.match, line, pos);
          if (m && (bestMatch === null || m.index < bestMatch.index)) {
            bestMatch = m;
            bestKind = 'match';
            bestRule = rule;
          }
        } else if (rule.begin !== undefined) {
          const m = execWithIndices(rule.begin, line, pos);
          if (m && (bestMatch === null || m.index < bestMatch.index)) {
            bestMatch = m;
            bestKind = 'begin';
            bestRule = rule;
          }
        }
      }

      if (!bestMatch) {
        if (pos < line.length) {
          tokens.push({
            start: pos,
            end: line.length,
            text: line.slice(pos),
            scopes: [...top.scopes],
          });
        }
        break;
      }

      if (bestMatch.index > pos) {
        tokens.push({
          start: pos,
          end: bestMatch.index,
          text: line.slice(pos, bestMatch.index),
          scopes: [...top.scopes],
        });
      }

      const matchEnd = bestMatch.index + bestMatch[0].length;

      if (bestKind === 'end') {
        if (bestRule.endCaptures && bestMatch[0].length > 0) {
          emitCaptures(tokens, bestMatch, bestRule.endCaptures, top.scopes);
        } else if (bestMatch[0].length > 0) {
          tokens.push({
            start: bestMatch.index,
            end: matchEnd,
            text: bestMatch[0],
            scopes: [...top.scopes],
          });
        }
        stateStack.pop();
        if (matchEnd === pos && bestMatch[0].length === 0) {
          // End matched `$` at end of line
          break;
        }
        pos = matchEnd;
      } else if (bestKind === 'match') {
        const scopes = bestRule.name ? [...top.scopes, bestRule.name] : [...top.scopes];
        const emitted = emitCaptures(tokens, bestMatch, bestRule.captures, scopes);
        if (!emitted && bestMatch[0].length > 0) {
          tokens.push({
            start: bestMatch.index,
            end: matchEnd,
            text: bestMatch[0],
            scopes,
          });
        }
        pos = Math.max(matchEnd, pos + 1);
      } else if (bestKind === 'begin') {
        const nextScopes = bestRule.name ? [...top.scopes, bestRule.name] : [...top.scopes];
        const emitted = emitCaptures(tokens, bestMatch, bestRule.beginCaptures, nextScopes);
        if (!emitted && bestMatch[0].length > 0) {
          tokens.push({
            start: bestMatch.index,
            end: matchEnd,
            text: bestMatch[0],
            scopes: nextScopes,
          });
        }
        stateStack.push({
          rule: bestRule,
          rules: bestRule.patterns ? expandPatterns(bestRule.patterns) : [],
          scopes: nextScopes,
        });
        pos = matchEnd;
      }
    }

    allLineTokens.push(tokens);
  }

  return allLineTokens;
}

function assertTokenScope(lineSource, needle, expectedScope, occurrence = 0) {
  const [tokens] = tokenizeSource(lineSource);
  let startIdx = -1;
  let searchFrom = 0;
  for (let k = 0; k <= occurrence; k += 1) {
    startIdx = lineSource.indexOf(needle, searchFrom);
    assert.notEqual(
      startIdx,
      -1,
      `Needle ${JSON.stringify(needle)} (occurrence ${occurrence}) not found in ${JSON.stringify(lineSource)}`,
    );
    searchFrom = startIdx + needle.length;
  }
  const endIdx = startIdx + needle.length;
  const overlapping = tokens.filter((t) => t.end > startIdx && t.start < endIdx);
  assert.ok(
    overlapping.length > 0 &&
      overlapping[0].start <= startIdx &&
      overlapping[overlapping.length - 1].end >= endIdx,
    `Tokens do not cover ${JSON.stringify(needle)} [${startIdx}..${endIdx}) in:\n  ${lineSource}\nTokens:\n  ${JSON.stringify(tokens, null, 2)}`,
  );
  for (const tok of overlapping) {
    assert.ok(
      tok.scopes.includes(expectedScope),
      `Expected ${JSON.stringify(needle)} in ${JSON.stringify(lineSource)} to include scope ${JSON.stringify(expectedScope)}, got ${JSON.stringify(tok.scopes)} (token=${JSON.stringify(tok.text)})`,
    );
  }
}

function assertTokenNotScope(lineSource, needle, forbiddenScope) {
  const [tokens] = tokenizeSource(lineSource);
  const startIdx = lineSource.indexOf(needle);
  assert.notEqual(startIdx, -1, `Needle ${JSON.stringify(needle)} not found in ${lineSource}`);
  const endIdx = startIdx + needle.length;
  const overlapping = tokens.filter((t) => t.end > startIdx && t.start < endIdx);
  for (const tok of overlapping) {
    assert.ok(
      !tok.scopes.includes(forbiddenScope),
      `Expected ${JSON.stringify(needle)} in ${JSON.stringify(lineSource)} NOT to have scope ${JSON.stringify(forbiddenScope)}, got ${JSON.stringify(tok.scopes)} (token=${JSON.stringify(tok.text)})`,
    );
  }
}

// ---------------------------------------------------------------------------
// 4. TextMate Highlighting Oracle Assertions (Waves 1-18)
// ---------------------------------------------------------------------------

// Comments & doc comments (exact 3-dash rule)
assertTokenScope('--- Doc comment with space', '--- Doc comment with space', 'comment.line.documentation.lust');
assertTokenScope('---@bindgen(name = "foo")', '@bindgen', 'entity.other.attribute-name.documentation.lust');
assertTokenScope('---@bindgen(name = "foo")', '---', 'comment.line.documentation.lust');
assertTokenScope('---', '---', 'comment.line.documentation.lust');
assertTokenScope('---- ordinary four-dash comment', '---- ordinary four-dash comment', 'comment.line.double-dash.lust');
assertTokenNotScope('---- ordinary four-dash comment', '----', 'comment.line.documentation.lust');
assertTokenScope('-- regular comment', '-- regular comment', 'comment.line.double-dash.lust');
assertTokenScope('# hash comment', '# hash comment', 'comment.line.number-sign.lust');
{
  const multi = tokenizeSource('--[[ block\ncomment ]]');
  assert.ok(multi[0][0].scopes.includes('comment.block.lust'));
  assert.ok(multi[1][0].scopes.includes('comment.block.lust'));
}

// Regex literals & strings
assertTokenScope('local r = regex!"^[a-z0-9_]+\\\\d+$"', 'regex', 'support.function.macro.lust');
assertTokenScope('local r = regex!"^[a-z0-9_]+\\\\d+$"', '!', 'keyword.operator.macro.lust');
assertTokenScope('local r = regex!"^[a-z0-9_]+\\\\d+$"', '^[a-z0-9_]+', 'string.regexp.lust');
assertTokenScope('local r = regex!"^[a-z0-9_]+\\\\d+$"', '\\\\', 'constant.character.escape.regexp.lust');
assertTokenScope("local r = regex!'abc'", 'abc', 'string.regexp.lust');
assertTokenScope('use "@std/regex" as re', '@std/regex', 'support.module.std.lust');
assertTokenScope('import math from "@std/math"', '@std/math', 'support.module.std.lust');
assertTokenScope('local s = "hello\\n\\u{1F600}"', '\\u{1F600}', 'constant.character.escape.lust');
assertTokenScope('local mail = "alex@lust-lang.org"', 'alex@lust-lang.org', 'string.quoted.double.lust');
assertTokenNotScope('local mail = "alex@lust-lang.org"', '@lust', 'entity.other.attribute-name.lust');

// @ attributes & @std/... imports
assertTokenScope('use @std/sync.{Queue}', '@std/sync', 'support.module.std.lust');
assertTokenScope('use @std/sync.{Queue}', 'Queue', 'support.type.lust');
assertTokenScope('pub use @std/json', 'pub', 'storage.modifier.lust');
assertTokenScope('pub use @std/json', '@std/json', 'support.module.std.lust');
assertTokenScope('use @std/test.clock', '@std/test.clock', 'support.module.std.lust');
assertTokenScope('@derive(Debug, Clone)', '@', 'punctuation.definition.attribute.lust');
assertTokenScope('@derive(Debug, Clone)', 'derive', 'entity.other.attribute-name.lust');
assertTokenScope('@test', 'test', 'entity.other.attribute-name.lust');
assertTokenScope('    id: int @min(1) @max(100)', 'min', 'entity.other.attribute-name.lust');
assertTokenScope('    email: string @email', 'email', 'entity.other.attribute-name.lust', 1);

// Resource & effect keywords
for (const kw of ['defer', 'errdefer', 'scoped', 'effect', 'handle', 'with', 'perform', 'resume', 'pure']) {
  const sample = kw === 'effect' ? 'local j = effect do' : `${kw} x`;
  assertTokenScope(sample, kw, 'keyword.control.effect.lust');
}
assertTokenScope('handle { let a = perform("Ask", [1]) } with { Ask(x) => resume(x + 1) }', 'handle', 'keyword.control.effect.lust');
assertTokenScope('handle { let a = perform("Ask", [1]) } with { Ask(x) => resume(x + 1) }', 'perform', 'keyword.control.effect.lust');
assertTokenScope('handle { let a = perform("Ask", [1]) } with { Ask(x) => resume(x + 1) }', 'with', 'keyword.control.effect.lust');
assertTokenScope('handle { let a = perform("Ask", [1]) } with { Ask(x) => resume(x + 1) }', 'resume', 'keyword.control.effect.lust');
assertTokenScope('handle { let a = perform("Ask", [1]) } with { Ask(x) => resume(x + 1) }', '=>', 'keyword.operator.arrow.lust');

// Declarations (schema, type, effect, service, impl, test/suite/describe, import)
assertTokenScope('schema User', 'schema', 'storage.type.schema.lust');
assertTokenScope('schema User', 'User', 'entity.name.type.schema.lust');
assertTokenScope('type UserId = int', 'type', 'storage.type.alias.lust');
assertTokenScope('type UserId = int', 'UserId', 'entity.name.type.alias.lust');
assertTokenScope('type Boxed<T> = Option<T>', 'type', 'storage.type.alias.lust');
assertTokenScope('type Boxed<T> = Option<T>', 'Boxed', 'entity.name.type.alias.lust');
assertTokenScope('type Boxed<T> = Option<T>', 'T', 'entity.name.type.lust');
assertTokenScope('effect Console', 'effect', 'storage.type.effect.lust');
assertTokenScope('effect Console', 'Console', 'entity.name.type.effect.lust');
assertTokenScope('service NetworkService: Network', 'service', 'storage.type.service.lust');
assertTokenScope('service NetworkService: Network', 'NetworkService', 'entity.name.type.service.lust');
assertTokenScope('impl Display for User', 'impl', 'storage.type.impl.lust');
assertTokenScope('impl Display for User', 'Display', 'entity.name.type.trait.lust');
assertTokenScope('impl Display for User', 'for', 'keyword.control.lust');
assertTokenScope('impl Display for User', 'User', 'entity.name.type.lust');
assertTokenScope('test "basic arithmetic" do', 'test', 'keyword.control.test.lust');
assertTokenScope('test.skip("ci") "redis probe" do', 'test', 'keyword.control.test.lust');
assertTokenScope('test.skip("ci") "redis probe" do', 'skip', 'entity.other.attribute-name.test.lust');
assertTokenScope('test.timeout(3000) "deadline" do', 'timeout', 'entity.other.attribute-name.test.lust');
assertTokenScope('test.only "focused" do', 'only', 'entity.other.attribute-name.test.lust');
assertTokenScope('test smoke_check do', 'test', 'keyword.control.test.lust');
assertTokenScope('test smoke_check do', 'smoke_check', 'entity.name.function.test.lust');
assertTokenScope('suite "Checkout" do', 'suite', 'keyword.control.test.lust');
assertTokenScope('describe "Effects" do', 'describe', 'keyword.control.test.lust');
assertTokenScope('before_each do', 'before_each', 'keyword.control.test.lust');
assertTokenScope('after_each do', 'after_each', 'keyword.control.test.lust');
assertTokenScope('before_all do', 'before_all', 'keyword.control.test.lust');
assertTokenScope('after_all do', 'after_all', 'keyword.control.test.lust');
assertTokenScope('import math from "@std/math"', 'import', 'keyword.control.import.lust');
assertTokenScope('import math from "@std/math"', 'from', 'keyword.control.import.lust');

// Contextual negative checks (must NOT be mis-highlighted as keywords/declarations)
assertTokenNotScope('local schema = 123', 'schema', 'storage.type.schema.lust');
assertTokenNotScope('local type = 456', 'type', 'storage.type.alias.lust');
assertTokenNotScope('local type = 456', 'type', 'support.function.builtin.lust');
assertTokenScope('string.match(s, "a")', 'string', 'support.module.stdlib.lust');
assertTokenScope('string.match(s, "a")', 'match', 'entity.name.function.call.lust');
assertTokenNotScope('string.match(s, "a")', 'match', 'keyword.control.conditional.lust');
assertTokenScope('task.resume(t)', 'task', 'support.module.stdlib.lust');
assertTokenScope('task.resume(t)', 'resume', 'entity.name.function.call.lust');
assertTokenNotScope('task.resume(t)', 'resume', 'keyword.control.effect.lust');
assertTokenScope('test.clock_now_ms()', 'test', 'support.module.stdlib.lust');
assertTokenNotScope('test.clock_now_ms()', 'test', 'keyword.control.test.lust');
assertTokenScope('test(42)', 'test', 'entity.name.function.call.lust');
assertTokenNotScope('test(42)', 'test', 'keyword.control.test.lust');

// Modifiers, storage, type operators, logical operators
for (const mod of ['pub', 'priv', 'weak', 'ref', 'local', 'let', 'const', 'mut', 'module', 'extern', 'use', 'export']) {
  const sample = mod === 'module' ? 'module' : `${mod} x`;
  assertTokenScope(sample, mod, 'storage.modifier.lust');
}
for (const op of ['typeof', 'is', 'as']) {
  assertTokenScope(`x ${op} User`, op, 'keyword.operator.type.lust');
}
for (const op of ['and', 'or', 'not']) {
  assertTokenScope(`a ${op} b`, op, 'keyword.operator.logical.lust');
}
assertTokenScope('match x', 'match', 'keyword.control.conditional.lust');
assertTokenScope('case Ask(v) => v', 'case', 'keyword.control.conditional.lust');

// Numeric literals (octal, hex, binary, underscore separators, scientific floats, range disambiguation)
assertTokenScope('local o = 0o755', '0o755', 'constant.numeric.octal.lust');
assertTokenScope('local o2 = 0O7_5_5', '0O7_5_5', 'constant.numeric.octal.lust');
assertTokenScope('local h = 0xDEAD_BEEF', '0xDEAD_BEEF', 'constant.numeric.hex.lust');
assertTokenScope('local b = 0b1010_0101', '0b1010_0101', 'constant.numeric.binary.lust');
assertTokenScope('local n = 1_000_000', '1_000_000', 'constant.numeric.integer.lust');
assertTokenScope('local f1 = 3.14_15e-2', '3.14_15e-2', 'constant.numeric.float.lust');
assertTokenScope('local f2 = 1e10', '1e10', 'constant.numeric.float.lust');
assertTokenScope('local f3 = 2_5E+4', '2_5E+4', 'constant.numeric.float.lust');
assertTokenScope('for i in 1..10 do', '1', 'constant.numeric.integer.lust');
assertTokenScope('for i in 1..10 do', '..', 'keyword.operator.range.lust');
assertTokenScope('for i in 1..10 do', '10', 'constant.numeric.integer.lust');

// Operators (pipe, arrows, scope ::, floor div, shifts, xor, tilde, compound assignments, !, ?)
assertTokenScope('local v = x |> f |> g', '|>', 'keyword.operator.pipe.lust');
assertTokenScope('local fn = function(x: int) -> int', '->', 'keyword.operator.arrow.lust');
assertTokenScope('Ask(x) => resume(x)', '=>', 'keyword.operator.arrow.lust');
assertTokenScope('local v = Mod::item', '::', 'keyword.operator.namespace.lust');
assertTokenScope('local q = a // b', '//', 'keyword.operator.arithmetic.lust');
assertTokenScope('local s1 = a << 2', '<<', 'keyword.operator.bitwise.lust');
assertTokenScope('local s2 = a >> 2', '>>', 'keyword.operator.bitwise.lust');
assertTokenScope('local xr = a ^^ b', '^^', 'keyword.operator.bitwise.lust');
assertTokenScope('local nt = ~a', '~', 'keyword.operator.bitwise.lust');
assertTokenScope('function f(): string ! HttpError ~ Network', '!', 'keyword.operator.error.lust');
assertTokenScope('function f(): string ! HttpError ~ Network', '~', 'keyword.operator.bitwise.lust');
assertTokenScope('local n: string? = get()?', '?', 'keyword.operator.optional.lust', 0);
assertTokenScope('local n: string? = get()?', '?', 'keyword.operator.optional.lust', 1);
for (const op of ['+=', '-=', '*=', '/=', '//=', '%=', '^=', '&=', '|=', '^^=', '<<=', '>>=']) {
  assertTokenScope(`x ${op} 1`, op, 'keyword.operator.assignment.compound.lust');
}

// Built-in types, constructors, single-letter type params vs ALL_CAPS constants
for (const ty of ['int', 'float', 'bool', 'string', 'usize', 'unknown', 'never']) {
  assertTokenScope(`local x: ${ty}`, ty, 'support.type.primitive.lust');
}
for (const ctor of ['Some', 'None', 'Ok', 'Err']) {
  assertTokenScope(`local x = ${ctor}`, ctor, 'support.type.constructor.lust');
}
for (const ty of [
  'Self', 'Option', 'Result', 'Array', 'Map', 'Table', 'Queue', 'Task',
  'Scope', 'Fiber', 'Schema', 'Regex', 'Weak', 'Effect', 'Infer', 'InferInput', 'InferOutput',
]) {
  assertTokenScope(`local x: ${ty}`, ty, 'support.type.lust');
}

// Single-letter type parameters T, K, V MUST be entity.name.type.lust (NOT constant.other.lust)
for (const tp of ['T', 'K', 'V']) {
  assertTokenScope(`function f<${tp}>(x: ${tp}): ${tp}`, tp, 'entity.name.type.lust');
  assertTokenNotScope(`function f<${tp}>(x: ${tp}): ${tp}`, tp, 'constant.other.lust');
}
// ALL_CAPS constants (>= 2 chars) MUST be constant.other.lust
for (const c of ['MAX_BUF', 'PI', 'HTTP_OK']) {
  assertTokenScope(`local x = ${c}`, c, 'constant.other.lust');
}

// Built-in functions
for (const fn of ['print', 'println', 'type', 'tostring', 'tonumber', 'assert', 'error', 'pcall', 'assert_eq']) {
  assertTokenScope(`${fn}(x)`, fn, 'support.function.builtin.lust');
}

// Stdlib namespaces
for (const mod of [
  'table', 'array', 'map', 'string', 'math', 'io', 'os', 'json',
  'msgpack', 'csv', 'regex', 'task', 'sync', 'time', 'rx', 'test',
]) {
  assertTokenScope(`${mod}.foo(1)`, mod, 'support.module.stdlib.lust');
}
assertTokenScope('use std.json', 'std', 'support.module.stdlib.lust');
assertTokenScope('use std.json', 'json', 'support.module.stdlib.lust');

// ---------------------------------------------------------------------------
// 5. Indentation rules oracle (language-configuration.json)
// ---------------------------------------------------------------------------
const incRe = new RegExp(langConfig.indentationRules.increaseIndentPattern);
const decRe = new RegExp(langConfig.indentationRules.decreaseIndentPattern);

const shouldIncrease = [
  'function add(a: int, b: int): int',
  'local function helper()',
  'pub function api(): int',
  'pure function calc(x: int): int',
  'struct Point',
  'enum Color',
  'trait Display',
  'impl Display for Point',
  'extern "C"',
  'schema User',
  'local schema Internal',
  'scoped do',
  '    scoped {',
  'effect do',
  '    local job = effect do',
  'effect Console',
  'handle do',
  'handle {',
  '    } with {',
  'with',
  '    defer do',
  '    defer {',
  '    errdefer do',
  '    errdefer {',
  'if x > 0 then',
  'elseif x < 0 then',
  'else',
  'while running do',
  'for i = 1, 10 do',
  'match value',
  'test "basic arithmetic" do',
  'test.timeout(3000) "deadline" do',
  'suite "Checkout" do',
  '    describe "Effects" do',
  '    before_each do',
  '    after_each do',
  '    before_all do',
  '    after_all do',
];

for (const line of shouldIncrease) {
  assert.ok(incRe.test(line), `increaseIndentPattern MUST match: ${JSON.stringify(line)}`);
}

const shouldNotIncrease = [
  'local x = 1',
  'defer close_socket(fd)',
  'errdefer record_failure_metric("fetch", url)',
  'local schema = 123',
  'local effect = 1',
  'if x then return 1 end',
  'local f = function(x: int): int return x end',
  'return x + 1',
];

for (const line of shouldNotIncrease) {
  assert.ok(!incRe.test(line), `increaseIndentPattern must NOT match: ${JSON.stringify(line)}`);
}

const shouldDecrease = [
  'end',
  '    end',
  'else',
  'elseif x == 2 then',
  'with',
  '    } with {',
  '}',
  ')',
  ']',
];

for (const line of shouldDecrease) {
  assert.ok(decRe.test(line), `decreaseIndentPattern MUST match: ${JSON.stringify(line)}`);
}

console.log(
  `PASS: vsc-extension grammar & language-configuration verified (${regexCount} regexes compiled cleanly, all Waves 1-18 TextMate & indentation assertions passed).`,
);
