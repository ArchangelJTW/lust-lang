---
name: coding-lust
description: Use this skill to become informed of the syntax and standard library of the Lust programming language
---

# Lust Language Guide & Skills Reference

Lust is a strongly typed, Lua-inspired scripting language implemented in Rust. It features strong static type checking, enum pattern matching, generic functions and types, trait-based polymorphism, cooperative multitasking, weak reference cycle safety, and a trace-based JIT.

---

## 1. Naming & Case Conventions

- **lowercase**: Primitive types and keywords (`int`, `float`, `string`, `bool`, `unknown`, `local`, `function`, `struct`, `enum`, `trait`, `impl`, `use`, `is`, `as`, `ref`).
- **PascalCase**: Nominal types, traits, and enum variants (`Option`, `Result`, `Some`, `None`, `Ok`, `Err`, `Array`, `Map`, `IndexError`, `TaskInfo`, `TaskStatus`, user types).
- **snake_case**: Variables, functions, methods, fields, and module namespaces.

---

## 2. Key Differences from Lua

1. **Static Typing**: Variables, function parameters, and returns support explicit type annotations (`local x: int = 10`, `function f(x: int): string`). Types are inferred if omitted.
2. **0-Based Array Indexing**: Arrays are 0-indexed (`arr[0]` is the first element).
3. **Non-Trapping Array Bracket Reads**: `arr[index]` returns `Result<T, IndexError>` rather than `nil` or throwing. `IndexError` contains `.index` and `.length`.
   - Use `arr:get(index)` or `array.get(arr, index)` to get an `Option<T>`.
   - Array index assignment (`arr[index] = val`) requires an existing slot and traps if out of bounds.
4. **Pattern Matching & Type Narrowing (`is`)**: Use `if val is Pattern then` or `if val is Type then` for testing and binding.
5. **Fallible Downcasting (`as`)**: `val as Type` downcasts from `unknown` and returns `Option<Type>` (e.g. `val as int is Some(n)`).
6. **No Spurious Truthiness**: Conditionals expect strict `bool` conditions or type guard expressions (`is`).
7. **Control Flow**: Includes `continue` in addition to `break`.
8. **Compound Assignment**: Supports `+=`, `-=`, `*=`, `/=`.
9. **Generics & Traits**: Supports generic functions, structs, enums, impl blocks, and nominal traits.
10. **Memory Safety & `ref`**: Struct fields marked `ref` create weak parent links (`Option<T>`) to prevent reference counting cycles.

---

## 3. Project Configuration (`lust-config.toml`)

Place `lust-config.toml` in the project root to configure modules and settings:

```toml
[settings]
stdlib_modules = ["io", "os"] # Enable optional standard library modules
jit = true                    # Enable JIT compiler (default: true)

[dependencies]
# Dependencies when using packages
# my-lib = "0.1.0"
# local-pkg = { path = "../local-pkg" }
```

---

## 4. Syntax & Language Constructs

### Variables & Types
```lust
-- Primitives
local a: int = 42
local b: float = 3.14159
local c: string = "hello"
local d: bool = true

-- Inferred types & mutability
local counter = 0
counter += 1

-- Unions & Unknown
local val: string | int = "text"
local mystery: unknown = 123

-- Arrays, Maps, and Tuples
local nums: Array<int> = [1, 2, 3]
local lookup: Map<string, int> = { ["apple"] = 10, orange = 20 }
local tuple: (string, int, bool) = ("item", 42, true)
```

### Control Flow
```lust
-- If-Elseif-Else
if x > 10 then
    println("large")
elseif x > 0 then
    println("small")
else
    println("zero or negative")
end

-- While Loop
local i = 0
while i < 5 do
    i += 1
end

-- Numeric For Loop: for var = start, end [, step] do
for j = 1, 10, 2 do
    if j == 5 then continue end
    if j > 7 then break end
    println(j)
end

-- For-In Loop: Arrays
for item in nums do
    println(item)
end

-- For-In Loop: Maps
for key, value in map.iter(lookup) do
    println(key .. ": " .. tostring(value))
end
```

### Functions, Lambdas & Multiple Returns
```lust
-- Function definition
function add(a: int, b: int): int
    return a + b
end

-- Generic function (no space before <)
function first<Item>(items: Array<Item>): Option<Item>
    return array.first(items)
end

-- Lambda / Anonymous function
local double = function(x: int): int
    return x * 2
end

-- Closures capture surrounding variables
local factor = 3
local scale = function(x: int): int
    return x * factor
end

-- Multiple return values (Tuple) and destructuring
function min_max(a: int, b: int): (int, int)
    if a < b then return a, b end
    return b, a
end

local low, high = min_max(10, 5)
```

### Structs, Impls & Methods
```lust
struct Point
    x: int
    y: int
end

impl Point
    -- Associated constructor function
    function new(x: int, y: int): Point
        return Point { x = x, y = y }
    end

    -- Instance method with 'self'
    function distance_squared(self): int
        return self.x * self.x + self.y * self.y
    end

    function translate(self, dx: int, dy: int): Point
        return Point { x = self.x + dx, y = self.y + dy }
    end
end

-- Instantiation & method invocation
local p = Point.new(3, 4)
println(p.x)                -- Field access: 3
println(p:distance_squared()) -- Method call: 25
```

### Reference Fields (`ref`)
Fields declared with `ref` are weak references typed as `Option<T>`:
```lust
struct Node
    name: string
    parent: ref Node          -- Automatically Option<Node>
    children: Array<Node>
end

impl Node
    function new(name: string): Node
        return Node { name = name, parent = Option.None, children = [] }
    end
end

local parent = Node.new("root")
local child = Node.new("child")
child.parent = parent         -- Can assign Node or Option<Node>
array.push(parent.children, child)

if child.parent is Some(p) then
    println("Parent: " .. p.name)
end
```

### Enums & Pattern Matching (`is`)
```lust
enum Status
    Pending
    Running
    Complete(int)
    Failed(string)
end

local s = Status.Complete(200)

if s is Status.Complete(code) then
    println("Done with: " .. tostring(code))
elseif s is Running then
    println("In progress")
elseif s is Status.Failed(msg) then
    println("Error: " .. msg)
end
```

### Built-in `Option` and `Result`
```lust
-- Option<T>: Option.Some(v) or Option.None
local opt: Option<int> = Option.Some(42)
if opt is Some(val) then
    println(val)
end
println(opt:unwrap_or(0))

-- Result<T, E>: Result.Ok(v) or Result.Err(e)
local res: Result<int, string> = Result.Ok(100)
if res is Ok(v) then
    println(v)
elseif res is Err(err) then
    println("Failed: " .. err)
end
```

### Traits & Polymorphism
```lust
trait Drawable
    function draw(self): string
end

struct Circle
    radius: float
end

impl Drawable for Circle
    function draw(self): string
        return "Circle(" .. tostring(self.radius) .. ")"
    end
end

-- Bounded generic function
function render<T: Drawable>(item: T): string
    return item:draw()
end

-- Dynamic dispatch via trait type
function render_dynamic(item: Drawable): string
    return item:draw()
end

-- Special ToString trait (enables automatic string concatenation via '..')
impl ToString for Circle
    function to_string(self): string
        return self:draw()
    end
end

local c = Circle { radius = 5.0 }
println("Drawing: " .. c) -- Uses ToString
```

### Modules & Imports
Files automatically map to dot-separated module paths relative to the source root (e.g. `lib/math/vector.lust` -> `lib.math.vector`).

```lust
-- In lib/math.lust:
struct Point
    x: int
    y: int
end

function add(a: int, b: int): int
    return a + b
end

-- In main.lust:
use lib.math as math
use lib.math.Point
use lib.math.{add, Point as MathPoint}
use lib.math.*
```

---

## 5. Standard Library Complete API Reference

### Global Built-in Functions
- `print(...)`: Prints values separated by tabs without a trailing newline.
- `println(...)`: Prints values separated by tabs with a trailing newline.
- `type(val)`: Returns the runtime type name as a string (`"int"`, `"float"`, `"string"`, `"bool"`, `"array"`, `"map"`, `"struct"`, `"enum"`, `"function"`, etc.).
- `tostring(val)`: Converts any value to a string.
- `tonumber(val [, base])`: Converts a string/int/float/bool to number.
- `error(message)`: Raises a runtime error with `message`.
- `assert(cond [, message])`: Asserts that `cond` is truthy; raises error otherwise.
- `select(index_or_hash, ...)`: Returns argument count for `"#"` or returns slice starting at 1-based index.
- `random([m [, n]])`: Random float in `[0, 1)`, or integer in `[1, m]` or `[m, n]`.
- `randomseed(seed)`: Seeds the random number generator.
- `unpack(arr [, i [, j]])`: Unpacks array elements into multiple return values.
- `pairs(map_or_table)`: Returns iterator over key/value pairs.
- `ipairs(arr_or_table)`: Returns iterator over index/value pairs.

---

### Built-in Methods

#### `Option<T>`
- `opt:is_some(): bool`
- `opt:is_none(): bool`
- `opt:unwrap(): T` (Panics if None)
- `opt:unwrap_or(default: T): T`

#### `Result<T, E>`
- `res:is_ok(): bool`
- `res:is_err(): bool`
- `res:unwrap(): T` (Panics if Err)
- `res:unwrap_or(default: T): T`

#### `Iterator`
- `iter:iter(): Iterator`
- `iter:next(): Option<unknown>`

---

### `array` Module
- `array.len<T>(arr: Array<T>): int`
- `array.is_empty<T>(arr: Array<T>): bool`
- `array.get<T>(arr: Array<T>, index: int): Option<T>`
- `array.first<T>(arr: Array<T>): Option<T>`
- `array.last<T>(arr: Array<T>): Option<T>`
- `array.push<T>(arr: Array<T>, val: T): ()`
- `array.pop<T>(arr: Array<T>): Option<T>`
- `array.insert<T>(arr: Array<T>, index: int, val: T): ()`
- `array.remove<T>(arr: Array<T>, index: int): Option<T>`
- `array.clear<T>(arr: Array<T>): ()`
- `array.slice<T>(arr: Array<T>, start: int, end: int): Array<T>`
- `array.concat<T>(arr: Array<T> [, sep: string]): string`
- `array.sort<T>(arr: Array<T> [, comp_fn]): ()`
- `array.reverse<T>(arr: Array<T>): ()`
- `array.contains<T>(arr: Array<T>, val: T): bool`
- `array.map<T>(arr: Array<T>, fn: function(T): U): Array<U>`
- `array.filter<T>(arr: Array<T>, fn: function(T): bool): Array<T>`
- `array.reduce<T>(arr: Array<T>, initial: U, fn: function(U, T): U): U`
- `array.iter<T>(arr: Array<T>): Iterator`

---

### `map` Module
- `map.len<K, V>(m: Map<K, V>): int`
- `map.is_empty<K, V>(m: Map<K, V>): bool`
- `map.get<K, V>(m: Map<K, V>, key: K): Option<V>`
- `map.set<K, V>(m: Map<K, V>, key: K, val: V): ()`
- `map.has<K, V>(m: Map<K, V>, key: K): bool`
- `map.delete<K, V>(m: Map<K, V>, key: K): Option<V>`
- `map.clear<K, V>(m: Map<K, V>): ()`
- `map.keys<K, V>(m: Map<K, V>): Array<K>`
- `map.values<K, V>(m: Map<K, V>): Array<V>`
- `map.iter<K, V>(m: Map<K, V>): Iterator` (yields `key, value`)

---

### `string` Module
- `string.len(s: string): int`
- `string.sub(s: string, i: int [, j: int]): string` (1-based indices, negative counts from end)
- `string.lower(s: string): string`
- `string.upper(s: string): string`
- `string.byte(s: string [, i [, j]]): unknown`
- `string.char(byte: int...): string`
- `string.find(s: string, pattern: string [, init [, plain]]): unknown`
- `string.match(s: string, pattern: string [, init]): unknown`
- `string.gsub(s: string, pattern: string, repl: unknown [, n]): unknown`
- `string.format(fmt: string, ...): string`
- `string.rep(s: string, n: int [, sep: string]): string`
- `string.reverse(s: string): string`
- `string.split(s: string, delimiter: string): Array<string>`
- `string.trim(s: string): string`
- `string.trim_start(s: string): string`
- `string.trim_end(s: string): string`
- `string.replace(s: string, from: string, to: string): string`
- `string.starts_with(s: string, prefix: string): bool`
- `string.ends_with(s: string, suffix: string): bool`
- `string.contains(s: string, substring: string): bool`
- `string.is_empty(s: string): bool`
- `string.chars(s: string): Array<string>`
- `string.lines(s: string): Array<string>`

---

### `math` Module
- **Constants**:
  - `math.pi`: float (`3.141592653589793`)
  - `math.huge`: float (`infinity`)
  - `math.maxinteger`: int
  - `math.mininteger`: int
- **Functions**:
  - `math.abs(x: number): number`
  - `math.floor(x: number): int`
  - `math.ceil(x: number): int`
  - `math.round(x: number): number`
  - `math.sqrt(x: float): float`
  - `math.sin(x: float): float`, `math.cos(x: float): float`, `math.tan(x: float): float`
  - `math.asin(x: float): float`, `math.acos(x: float): float`, `math.atan(x: float): float`
  - `math.atan2(y: float, x: float): float`
  - `math.min(x, y, ...): number`
  - `math.max(x, y, ...): number`
  - `math.clamp(x, min, max): number`
  - `math.random([m [, n]]): number`
  - `math.randomseed(seed: int): ()`
  - `math.deg(rad: float): float`
  - `math.rad(deg: float): float`
  - `math.exp(x: float): float`
  - `math.log(x: float [, base: float]): float`
  - `math.tointeger(x: unknown): Option<int>`
  - `math.tofloat(x: unknown): Option<float>`

---

### `io` Module *(Requires `stdlib_modules = ["io"]` in `lust-config.toml`)*
- `io.read_file(path: string): Result<string, string>`
- `io.read_file_bytes(path: string): Result<Array<int>, string>`
- `io.write_file(path: string, contents: unknown): Result<(), string>`
- `io.read_stdin(): Result<string, string>`
- `io.read_line(): Result<string, string>`
- `io.write_stdout(value: unknown): Result<(), string>`

---

### `os` Module *(Requires `stdlib_modules = ["os"]` in `lust-config.toml`)*
- `os.time(): float` (Unix timestamp in seconds with sub-second precision)
- `os.sleep(seconds: float): Result<(), string>`
- `os.create_file(path: string): Result<(), string>`
- `os.create_dir(path: string): Result<(), string>`
- `os.remove_file(path: string): Result<(), string>`
- `os.remove_dir(path: string): Result<(), string>`
- `os.rename(from: string, to: string): Result<(), string>`

---

### `task` Module (Cooperative Tasks)
- `task.create(fn: function): Task`: Creates a suspended task.
- `task.run(fn: function): Task`: Spawns and starts a task immediately.
- `task.yield(value: unknown): unknown`: Yields execution from current task.
- `task.resume(t: Task): TaskInfo`: Resumes a task and returns status info.
- `task.status(t: Task): TaskStatus`: Returns current `TaskStatus` (`Ready`, `Running`, `Yielded`, `Completed`, `Failed`, `Stopped`).
- `task.info(t: Task): TaskInfo`: Inspects task state (`info.state: TaskStatus`, `info.last_yield: Option<unknown>`, `info.last_result: Option<unknown>`, `info.error: Option<string>`).
- `task.stop(t: Task): bool`: Stops a running task.
- `task.restart(t: Task): TaskInfo`: Restarts a completed task.
- `task.current(): Option<Task>`: Returns the currently executing task handle.

---

### `lua` Module (Lua Interop)
- `lua.to_value(v: unknown): LuaValue`
- `lua.unwrap(v: unknown): unknown`
- `lua.require(module_name: string): unknown`
- `lua.table(): LuaTable`
- `lua.setmetatable(tbl: LuaValue, meta: LuaValue): LuaValue`
- `lua.getmetatable(tbl: LuaValue): LuaValue`
- `lua.is_truthy(v: unknown): bool`
