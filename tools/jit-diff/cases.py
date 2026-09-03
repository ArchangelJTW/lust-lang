"""Case definitions for the Lust JIT differential tester.

Every case is a self-contained Lust program that prints exactly one integer on
stdout.  The expected value is computed here in Python using the same arithmetic
so that we can distinguish three separate failure classes:

  * the JIT disagrees with the interpreter  (JIT bug)
  * the interpreter disagrees with Python   (interpreter / compiler bug)
  * both disagree with Python               (front-end bug)

Cases only ever print integers.  Float workloads accumulate in a float and then
scale + truncate via `:to_int()`, which is exactly reproducible in Python with
`math.trunc`, so we never have to compare formatted floating point text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator

# Loop variables used by the generated loop nests.  Deliberately unlikely names:
# the body snippets below declare their own locals, and a collision silently
# turns a JIT test into a type error.
LV = ("iter_a", "iter_b", "iter_c")


@dataclass(frozen=True)
class Case:
    name: str
    source: str
    expected: int
    tags: tuple[str, ...] = field(default=())


# Trip counts chosen around the JIT's own thresholds:
#   HOT_THRESHOLD      = 5    (iterations before a loop is traced)
#   LOOP_UNROLL_FACTOR = 32   (unroll width)
#   SIDE_EXIT_THRESHOLD= 10   (guard failures before a side trace)
#   MAX_TRACE_LENGTH   = 2000
TRIPS_1D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1000]
OUTER_2D = [1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 32, 100]
INNER_2D = [1, 2, 3, 4, 5, 6, 7, 8, 31, 32, 33]
TRIPS_3D = [1, 2, 3, 5, 6, 8]
# Trip counts for the per-operation sweep: dense around HOT_THRESHOLD so we can
# read off the exact iteration at which an operation starts to diverge.
HOT_SWEEP = [1, 2, 3, 4, 5, 6, 7, 8, 10, 16, 32, 100]


# ---------------------------------------------------------------------------
# Group 1: loop nesting depth vs trip count, with the simplest possible body.
#
# This is the core grid.  Body is a single integer increment so that anything
# that goes wrong can only be the loop machinery itself.
# ---------------------------------------------------------------------------


def loop_shapes() -> Iterator[Case]:
    a, b, c = LV

    for n in TRIPS_1D:
        yield Case(
            f"shape/for/d1/n{n}",
            f"local acc: int = 0\n"
            f"for {a} = 1, {n} do\n"
            f"  acc = acc + 1\n"
            f"end\n"
            f"println(acc)\n",
            n,
            ("shape", "for", "depth1"),
        )

    for n in TRIPS_1D:
        yield Case(
            f"shape/while/d1/n{n}",
            f"local acc: int = 0\n"
            f"local {a}: int = 1\n"
            f"while {a} <= {n} do\n"
            f"  acc = acc + 1\n"
            f"  {a} = {a} + 1\n"
            f"end\n"
            f"println(acc)\n",
            n,
            ("shape", "while", "depth1"),
        )

    for m in OUTER_2D:
        for n in INNER_2D:
            yield Case(
                f"shape/for/d2/m{m}_n{n}",
                f"local acc: int = 0\n"
                f"for {a} = 1, {m} do\n"
                f"  for {b} = 1, {n} do\n"
                f"    acc = acc + 1\n"
                f"  end\n"
                f"end\n"
                f"println(acc)\n",
                m * n,
                ("shape", "for", "depth2"),
            )

    for m in OUTER_2D:
        for n in INNER_2D:
            yield Case(
                f"shape/while/d2/m{m}_n{n}",
                f"local acc: int = 0\n"
                f"local {a}: int = 1\n"
                f"while {a} <= {m} do\n"
                f"  local {b}: int = 1\n"
                f"  while {b} <= {n} do\n"
                f"    acc = acc + 1\n"
                f"    {b} = {b} + 1\n"
                f"  end\n"
                f"  {a} = {a} + 1\n"
                f"end\n"
                f"println(acc)\n",
                m * n,
                ("shape", "while", "depth2"),
            )

    # Mixed for/while nesting: isolates whether the bug is per loop-kind.
    for m in OUTER_2D:
        for n in [3, 5, 32]:
            yield Case(
                f"shape/for_while/d2/m{m}_n{n}",
                f"local acc: int = 0\n"
                f"for {a} = 1, {m} do\n"
                f"  local {b}: int = 1\n"
                f"  while {b} <= {n} do\n"
                f"    acc = acc + 1\n"
                f"    {b} = {b} + 1\n"
                f"  end\n"
                f"end\n"
                f"println(acc)\n",
                m * n,
                ("shape", "mixed", "depth2"),
            )
            yield Case(
                f"shape/while_for/d2/m{m}_n{n}",
                f"local acc: int = 0\n"
                f"local {a}: int = 1\n"
                f"while {a} <= {m} do\n"
                f"  for {b} = 1, {n} do\n"
                f"    acc = acc + 1\n"
                f"  end\n"
                f"  {a} = {a} + 1\n"
                f"end\n"
                f"println(acc)\n",
                m * n,
                ("shape", "mixed", "depth2"),
            )

    for x in TRIPS_3D:
        for y in TRIPS_3D:
            for z in [1, 3, 5]:
                yield Case(
                    f"shape/for/d3/{x}_{y}_{z}",
                    f"local acc: int = 0\n"
                    f"for {a} = 1, {x} do\n"
                    f"  for {b} = 1, {y} do\n"
                    f"    for {c} = 1, {z} do\n"
                    f"      acc = acc + 1\n"
                    f"    end\n"
                    f"  end\n"
                    f"end\n"
                    f"println(acc)\n",
                    x * y * z,
                    ("shape", "for", "depth3"),
                )

    # Step values, descending loops, zero-trip loops.
    for start, stop, step in [
        (1, 100, 2),
        (0, 100, 3),
        (0, 64, 32),
        (100, 1, -1),
        (100, 1, -7),
        (5, 1, 1),  # zero-trip: start > stop with positive step
        (1, 1, 1),
    ]:
        count = len(range(start, stop + (1 if step > 0 else -1), step))
        yield Case(
            f"shape/for/step/{start}_{stop}_{step}",
            f"local acc: int = 0\n"
            f"for {a} = {start}, {stop}, {step} do\n"
            f"  acc = acc + 1\n"
            f"end\n"
            f"println(acc)\n",
            count,
            ("shape", "for", "step"),
        )

    # Loops long enough to exceed MAX_TRACE_LENGTH bookkeeping.
    for n in [1999, 2000, 2001, 4096, 100000]:
        yield Case(
            f"shape/for/d1/long_n{n}",
            f"local acc: int = 0\n"
            f"for {a} = 1, {n} do\n"
            f"  acc = acc + 1\n"
            f"end\n"
            f"println(acc)\n",
            n,
            ("shape", "for", "long"),
        )

    # Loop variable read inside the body (induction variable use).
    for n in [5, 32, 100, 1000]:
        yield Case(
            f"shape/for/indvar/n{n}",
            f"local acc: int = 0\n"
            f"for {a} = 1, {n} do\n"
            f"  acc = acc + {a}\n"
            f"end\n"
            f"println(acc)\n",
            n * (n + 1) // 2,
            ("shape", "for", "indvar"),
        )
    for m, n in [(20, 5), (5, 20), (32, 32), (100, 3)]:
        yield Case(
            f"shape/for/indvar_d2/m{m}_n{n}",
            f"local acc: int = 0\n"
            f"for {a} = 1, {m} do\n"
            f"  for {b} = 1, {n} do\n"
            f"    acc = acc + {b}\n"
            f"  end\n"
            f"end\n"
            f"println(acc)\n",
            m * (n * (n + 1) // 2),
            ("shape", "for", "indvar", "depth2"),
        )


# ---------------------------------------------------------------------------
# Group 2: body operations.
#
# Each op has an `expected` function of the total inner-iteration count, so the
# same op can be run at any shape and any trip count.
# ---------------------------------------------------------------------------


@dataclass
class Op:
    name: str
    decls: str
    body: str
    result: str
    expected: Callable[[int], int]
    tags: tuple[str, ...] = field(default=())


def _sim_int_mac(t: int) -> int:
    acc = 0
    for _ in range(t):
        acc = acc + 3
        acc = acc * 2 % 1000003
    return acc


def _sim_int_div(t: int) -> int:
    acc = 0
    for _ in range(t):
        acc = (acc + 7) // 2  # both operands non-negative, so // == truncation
    return acc


def _sim_float_mul(t: int) -> int:
    acc = 0.0
    for _ in range(t):
        acc = acc * 0.5 + 1.0
    return math.trunc(acc * 1000000.0)


def _sim_float_sqrt(t: int) -> int:
    acc = 0.0
    for _ in range(t):
        acc = acc + math.sqrt(2.0)
    return math.trunc(acc * 1000.0)


def _sim_float_div(t: int) -> int:
    acc = 1.0
    for k in range(1, t + 1):
        acc = acc + 1.0 / float(k)
    return math.trunc(acc * 1000000.0)


def _sim_arr_read(t: int) -> int:
    return sum((k % 8) + 1 for k in range(t))


def _sim_arr_read_float(t: int) -> int:
    acc = 0.0
    for k in range(t):
        acc = acc + float((k % 8) + 1)
    return math.trunc(acc * 1000.0)


def _sim_arr_write_float(t: int) -> int:
    cells = [0.0] * 8
    for k in range(t):
        cells[k % 8] = cells[k % 8] + 0.25
    return math.trunc(sum(cells) * 100.0)


_NESTED = [[1, 2], [3, 4], [5, 6], [7, 8]]


def OPS() -> list[Op]:
    ops: list[Op] = []

    # -- integer arithmetic ------------------------------------------------
    ops.append(Op("int_add", "local acc: int = 0\n", "acc = acc + 1\n", "acc", lambda t: t, ("int",)))
    ops.append(Op("int_sub", "local acc: int = 0\n", "acc = acc - 2\n", "acc", lambda t: -2 * t, ("int",)))
    ops.append(
        Op(
            "int_mac",
            "local acc: int = 0\n",
            "acc = acc + 3\nacc = acc * 2 % 1000003\n",
            "acc",
            _sim_int_mac,
            ("int",),
        )
    )
    ops.append(Op("int_div", "local acc: int = 0\n", "acc = (acc + 7) / 2\n", "acc", _sim_int_div, ("int",)))
    ops.append(
        Op(
            "int_mod",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nacc = acc + ctr % 7\n",
            "acc",
            lambda t: sum(k % 7 for k in range(1, t + 1)),
            ("int",),
        )
    )
    ops.append(
        Op(
            "int_neg_abs",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nlocal neg: int = 0 - ctr\nacc = acc + math.abs(neg)\n",
            "acc",
            lambda t: t * (t + 1) // 2,
            ("int", "module"),
        )
    )
    ops.append(
        Op(
            "int_minmax",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nacc = acc + math.min(ctr, 50)\n",
            "acc",
            lambda t: sum(min(k, 50) for k in range(1, t + 1)),
            ("int", "module"),
        )
    )

    # -- float arithmetic --------------------------------------------------
    ops.append(
        Op(
            "float_add",
            "local facc: float = 0.0\n",
            "facc = facc + 0.5\n",
            "(facc * 2.0) as int",
            lambda t: t,
            ("float",),
        )
    )
    ops.append(
        Op(
            "float_mul",
            "local facc: float = 0.0\n",
            "facc = facc * 0.5 + 1.0\n",
            "(facc * 1000000.0) as int",
            _sim_float_mul,
            ("float",),
        )
    )
    ops.append(
        Op(
            "float_sqrt",
            "local facc: float = 0.0\nlocal two: float = 2.0\n",
            "facc = facc + math.sqrt(two)\n",
            "(facc * 1000.0) as int",
            _sim_float_sqrt,
            ("float", "module"),
        )
    )
    ops.append(
        Op(
            "float_div",
            "local facc: float = 1.0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nfacc = facc + 1.0 / (ctr as float)\n",
            "(facc * 1000000.0) as int",
            _sim_float_div,
            ("float",),
        )
    )
    ops.append(
        Op(
            "float_floor",
            "local facc: float = 0.0\nlocal fv: float = 3.7\n",
            "facc = facc + math.floor(fv)\n",
            "facc as int",
            lambda t: 3 * t,
            ("float", "module"),
        )
    )

    # -- control flow ------------------------------------------------------
    ops.append(
        Op(
            "if_then",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nif ctr % 2 == 0 then\n  acc = acc + 1\nend\n",
            "acc",
            lambda t: sum(1 for k in range(1, t + 1) if k % 2 == 0),
            ("control",),
        )
    )
    ops.append(
        Op(
            "if_else",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nif ctr % 2 == 0 then\n  acc = acc + 2\nelse\n  acc = acc + 1\nend\n",
            "acc",
            lambda t: sum(2 if k % 2 == 0 else 1 for k in range(1, t + 1)),
            ("control",),
        )
    )
    ops.append(
        Op(
            "if_elseif_chain",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\n"
            "if ctr % 15 == 0 then\n  acc = acc + 4\n"
            "elseif ctr % 3 == 0 then\n  acc = acc + 3\n"
            "elseif ctr % 5 == 0 then\n  acc = acc + 2\n"
            "else\n  acc = acc + 1\nend\n",
            "acc",
            lambda t: sum(
                4 if k % 15 == 0 else 3 if k % 3 == 0 else 2 if k % 5 == 0 else 1
                for k in range(1, t + 1)
            ),
            ("control",),
        )
    )
    ops.append(
        Op(
            "continue_stmt",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nif ctr % 2 == 0 then\n  continue\nend\nacc = acc + 1\n",
            "acc",
            lambda t: sum(1 for k in range(1, t + 1) if k % 2 != 0),
            ("control",),
        )
    )
    # A rarely-taken branch: forces repeated guard failures / side traces.
    ops.append(
        Op(
            "cold_branch",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nif ctr % 37 == 0 then\n  acc = acc + 1000\nelse\n  acc = acc + 1\nend\n",
            "acc",
            lambda t: sum(1000 if k % 37 == 0 else 1 for k in range(1, t + 1)),
            ("control", "sideexit"),
        )
    )
    ops.append(
        Op(
            "and_or",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nif ctr > 10 and ctr < 90 then\n  acc = acc + 1\nend\n",
            "acc",
            lambda t: sum(1 for k in range(1, t + 1) if 10 < k < 90),
            ("control", "bool"),
        )
    )
    ops.append(
        Op(
            "bool_local",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nlocal hit: bool = ctr > 10 and ctr < 90\nif hit then\n  acc = acc + 1\nend\n",
            "acc",
            lambda t: sum(1 for k in range(1, t + 1) if 10 < k < 90),
            ("control", "bool"),
        )
    )
    ops.append(
        Op(
            "or_short_circuit",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nif ctr < 5 or ctr > 95 then\n  acc = acc + 1\nend\n",
            "acc",
            lambda t: sum(1 for k in range(1, t + 1) if k < 5 or k > 95),
            ("control", "bool"),
        )
    )

    # -- arrays ------------------------------------------------------------
    ops.append(
        Op(
            "arr_read_int",
            "local arr: Array<int> = [1, 2, 3, 4, 5, 6, 7, 8]\nlocal acc: int = 0\nlocal ctr: int = 0\n",
            "acc = acc + arr[ctr % 8]:unwrap()\nctr = ctr + 1\n",
            "acc",
            _sim_arr_read,
            ("array", "int"),
        )
    )
    ops.append(
        Op(
            "arr_read_int_const_index",
            "local arr: Array<int> = [1, 2, 3, 4, 5, 6, 7, 8]\nlocal acc: int = 0\n",
            "acc = acc + arr[3]:unwrap()\n",
            "acc",
            lambda t: 4 * t,
            ("array", "int"),
        )
    )
    ops.append(
        Op(
            "arr_read_float",
            "local arr: Array<float> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]\n"
            "local facc: float = 0.0\nlocal ctr: int = 0\n",
            "facc = facc + arr[ctr % 8]:unwrap()\nctr = ctr + 1\n",
            "(facc * 1000.0):to_int()",
            _sim_arr_read_float,
            ("array", "float"),
        )
    )
    ops.append(
        Op(
            "arr_write_int",
            "local arr: Array<int> = [0, 0, 0, 0, 0, 0, 0, 0]\nlocal ctr: int = 0\n",
            "arr[ctr % 8] = arr[ctr % 8]:unwrap() + 1\nctr = ctr + 1\n",
            "arr[0]:unwrap() + arr[1]:unwrap() + arr[2]:unwrap() + arr[3]:unwrap() + arr[4]:unwrap() + arr[5]:unwrap() + arr[6]:unwrap() + arr[7]:unwrap()",
            lambda t: t,
            ("array", "int"),
        )
    )
    ops.append(
        Op(
            "arr_write_float",
            "local arr: Array<float> = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\nlocal ctr: int = 0\n",
            "arr[ctr % 8] = arr[ctr % 8]:unwrap() + 0.25\nctr = ctr + 1\n",
            "((arr[0]:unwrap() + arr[1]:unwrap() + arr[2]:unwrap() + arr[3]:unwrap() + arr[4]:unwrap() + arr[5]:unwrap() + arr[6]:unwrap() + arr[7]:unwrap()) * 100.0) as int",
            _sim_arr_write_float,
            ("array", "float"),
        )
    )
    ops.append(
        Op(
            "arr_push",
            "local arr: Array<int> = []\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\narray.push(arr, ctr)\n",
            "array.len(arr)",
            lambda t: t,
            ("array", "alloc"),
        )
    )
    ops.append(
        Op(
            "arr_push_pop",
            "local arr: Array<int> = []\nlocal acc: int = 0\n",
            "array.push(arr, 1)\nlocal got = array.pop(arr)\nacc = acc + got:unwrap_or(0)\n",
            "acc",
            lambda t: t,
            ("array", "alloc", "option"),
        )
    )
    ops.append(
        Op(
            "arr_len",
            "local arr: Array<int> = [1, 2, 3, 4, 5, 6, 7, 8]\nlocal acc: int = 0\n",
            "acc = acc + array.len(arr)\n",
            "acc",
            lambda t: 8 * t,
            ("array",),
        )
    )
    ops.append(
        Op(
            "arr_get_option",
            "local arr: Array<int> = [1, 2, 3, 4, 5, 6, 7, 8]\nlocal acc: int = 0\nlocal ctr: int = 0\n",
            "local got = array.get(arr, ctr % 8)\nacc = acc + got:unwrap_or(0)\nctr = ctr + 1\n",
            "acc",
            _sim_arr_read,
            ("array", "option"),
        )
    )
    ops.append(
        Op(
            "arr_nested_index",
            "local grid: Array<Array<int>> = [[1, 2], [3, 4], [5, 6], [7, 8]]\n"
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "acc = acc + grid[ctr % 4]:unwrap()[ctr % 2]:unwrap()\nctr = ctr + 1\n",
            "acc",
            lambda t: sum(_NESTED[k % 4][k % 2] for k in range(t)),
            ("array", "nested"),
        )
    )
    ops.append(
        Op(
            "arr_for_in",
            "local arr: Array<int> = [1, 2, 3, 4, 5, 6, 7, 8]\nlocal acc: int = 0\n",
            "for elem in arr do\n  acc = acc + elem\nend\n",
            "acc",
            lambda t: 36 * t,
            ("array", "forin"),
        )
    )

    # -- structs -----------------------------------------------------------
    struct_decl = "struct Point\n  x: int,\n  y: int\nend\n"
    ops.append(
        Op(
            "struct_read",
            struct_decl + "local pt = Point { x = 3, y = 4 }\nlocal acc: int = 0\n",
            "acc = acc + pt.x + pt.y\n",
            "acc",
            lambda t: 7 * t,
            ("struct",),
        )
    )
    ops.append(
        Op(
            "struct_write",
            struct_decl + "local pt = Point { x = 0, y = 0 }\n",
            "pt.x = pt.x + 1\npt.y = pt.y + 2\n",
            "pt.x + pt.y",
            lambda t: 3 * t,
            ("struct",),
        )
    )
    ops.append(
        Op(
            "struct_alloc",
            struct_decl + "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nlocal pt = Point { x = ctr, y = 1 }\nacc = acc + pt.y\n",
            "acc",
            lambda t: t,
            ("struct", "alloc"),
        )
    )
    ops.append(
        Op(
            "struct_arr_read",
            struct_decl
            + "local pts: Array<Point> = []\n"
            "for seed = 1, 8 do\n  array.push(pts, Point { x = seed, y = 1 })\nend\n"
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "local pt = pts[ctr % 8]:unwrap()\nacc = acc + pt.x\nctr = ctr + 1\n",
            "acc",
            _sim_arr_read,
            ("struct", "array"),
        )
    )
    ops.append(
        Op(
            "struct_arr_write",
            struct_decl
            + "local pts: Array<Point> = []\n"
            "for seed = 1, 8 do\n  array.push(pts, Point { x = 0, y = 0 })\nend\n"
            "local ctr: int = 0\n",
            "local pt = pts[ctr % 8]:unwrap()\npt.x = pt.x + 1\nctr = ctr + 1\n",
            "pts[0]:unwrap().x + pts[1]:unwrap().x + pts[2]:unwrap().x + pts[3]:unwrap().x + pts[4]:unwrap().x + pts[5]:unwrap().x + pts[6]:unwrap().x + pts[7]:unwrap().x",
            lambda t: t,
            ("struct", "array"),
        )
    )
    ops.append(
        Op(
            "struct_method",
            "struct Counter\n  n: int\nend\n"
            "impl Counter\n"
            "  function bump(self): int\n    self.n = self.n + 1\n    return self.n\n  end\n"
            "end\n"
            "local ctr2 = Counter { n = 0 }\nlocal acc: int = 0\n",
            "acc = acc + ctr2:bump()\n",
            "acc",
            lambda t: t * (t + 1) // 2,
            ("struct", "method"),
        )
    )

    # -- calls -------------------------------------------------------------
    ops.append(
        Op(
            "func_call",
            "function addone(v: int): int\n  return v + 1\nend\nlocal acc: int = 0\n",
            "acc = addone(acc)\n",
            "acc",
            lambda t: t,
            ("call",),
        )
    )
    ops.append(
        Op(
            "func_call_early_return",
            "function pick(v: int): int\n  if v % 2 == 0 then\n    return 2\n  end\n  return 1\nend\n"
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nacc = acc + pick(ctr)\n",
            "acc",
            lambda t: sum(2 if k % 2 == 0 else 1 for k in range(1, t + 1)),
            ("call", "control"),
        )
    )
    ops.append(
        Op(
            "closure_call",
            "local step: int = 3\nlocal fn = function(v: int): int\n  return v + step\nend\nlocal acc: int = 0\n",
            "acc = fn(acc)\n",
            "acc",
            lambda t: 3 * t,
            ("call", "closure"),
        )
    )
    ops.append(
        Op(
            "recursive_call",
            "function fib(n: int): int\n  if n < 2 then\n    return n\n  end\n  return fib(n - 1) + fib(n - 2)\nend\n"
            "local acc: int = 0\n",
            "acc = acc + fib(10)\n",
            "acc",
            lambda t: 55 * t,
            ("call", "recursion"),
        )
    )

    # -- strings and maps --------------------------------------------------
    ops.append(
        Op(
            "str_concat",
            'local buf: string = ""\n',
            'buf = buf .. "x"\n',
            "string.len(buf)",
            lambda t: t,
            ("string", "alloc"),
        )
    )
    ops.append(
        Op(
            "str_methods",
            'local text: string = "hello world"\nlocal acc: int = 0\n',
            'if string.contains(text, "world") then\n  acc = acc + string.len(text)\nend\n',
            "acc",
            lambda t: 11 * t,
            ("string",),
        )
    )
    ops.append(
        Op(
            "str_tostring",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nacc = acc + string.len(tostring(ctr))\n",
            "acc",
            lambda t: sum(len(str(k)) for k in range(1, t + 1)),
            ("string",),
        )
    )
    ops.append(
        Op(
            "map_set_get",
            "local tbl: Map<string, int> = {}\nlocal acc: int = 0\n",
            'map.set(tbl, "k", acc + 1)\nlocal got = map.get(tbl, "k")\nacc = got:unwrap_or(0)\n',
            "acc",
            lambda t: t,
            ("map", "option"),
        )
    )
    ops.append(
        Op(
            "map_grow",
            "local tbl: Map<string, int> = {}\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\nmap.set(tbl, tostring(ctr), ctr)\n",
            "map.len(tbl)",
            lambda t: t,
            ("map", "alloc"),
        )
    )

    # -- Option / enum -----------------------------------------------------
    ops.append(
        Op(
            "option_match",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\n"
            "local opt: Option<int> = Option.Some(ctr)\n"
            "if opt is Some(v) then\n  acc = acc + v\nend\n",
            "acc",
            lambda t: t * (t + 1) // 2,
            ("option", "enum"),
        )
    )
    ops.append(
        Op(
            "option_none_mix",
            "local acc: int = 0\nlocal ctr: int = 0\n",
            "ctr = ctr + 1\n"
            "local opt: Option<int> = Option.None\n"
            "if ctr % 2 == 0 then\n  opt = Option.Some(1)\nend\n"
            "acc = acc + opt:unwrap_or(0)\n",
            "acc",
            lambda t: sum(1 for k in range(1, t + 1) if k % 2 == 0),
            ("option", "enum", "sideexit"),
        )
    )

    return ops


def _indent(body: str, spaces: int) -> str:
    pad = " " * spaces
    return "".join(pad + ln + "\n" for ln in body.rstrip("\n").split("\n"))


def _shape_d1(body: str, n: int) -> str:
    return f"for {LV[0]} = 1, {n} do\n{_indent(body, 2)}end\n"


def _shape_d2(body: str, m: int, n: int) -> str:
    return (
        f"for {LV[0]} = 1, {m} do\n"
        f"  for {LV[1]} = 1, {n} do\n"
        f"{_indent(body, 4)}"
        f"  end\n"
        f"end\n"
    )


def body_ops() -> Iterator[Case]:
    """Every op at three shapes, all with 100 total inner iterations."""
    shapes = [
        ("d1", lambda b: _shape_d1(b, 100)),
        ("d2", lambda b: _shape_d2(b, 20, 5)),
        ("d2r", lambda b: _shape_d2(b, 5, 20)),
    ]
    for op in OPS():
        for shape_name, wrap in shapes:
            src = op.decls + wrap(op.body) + f"println({op.result})\n"
            yield Case(
                f"op/{op.name}/{shape_name}",
                src,
                op.expected(100),
                ("op", op.name, shape_name) + op.tags,
            )


def hot_sweep() -> Iterator[Case]:
    """Every op in a *single* loop, swept across trip counts.

    Any divergence here is a JIT bug that does not need nesting to trigger, and
    the trip count at which it first appears points straight at the threshold
    that mishandles it.
    """
    for op in OPS():
        for n in HOT_SWEEP:
            src = op.decls + _shape_d1(op.body, n) + f"println({op.result})\n"
            yield Case(
                f"hot/{op.name}/n{n}",
                src,
                op.expected(n),
                ("hot", op.name, "depth1") + op.tags,
            )


# ---------------------------------------------------------------------------
# Group 3: the specific reductions already known to fail, kept as named
# regressions so they stay visible at the top of the report.
# ---------------------------------------------------------------------------


def regressions() -> Iterator[Case]:
    yield Case(
        "regression/nested_offbyone",
        "local acc: int = 0\n"
        "for s = 1, 3 do\n"
        "  for i = 1, 3 do\n"
        "    acc = acc + 1\n"
        "  end\n"
        "end\n"
        "println(acc)\n",
        9,
        ("regression",),
    )
    yield Case(
        "regression/nested_hang",
        "local acc: int = 0\n"
        "for s = 1, 7 do\n"
        "  for i = 1, 3 do\n"
        "    acc = acc + 1\n"
        "  end\n"
        "end\n"
        "println(acc)\n",
        21,
        ("regression",),
    )
    yield Case(
        "regression/nested_array_int",
        "local a: Array<int> = [1, 2, 3, 4, 5]\n"
        "local acc: int = 0\n"
        "for s = 1, 20 do\n"
        "  for i = 0, 4 do\n"
        "    acc = acc + a[i]:unwrap()\n"
        "  end\n"
        "end\n"
        "println(acc)\n",
        300,
        ("regression",),
    )
    yield Case(
        "regression/nested_array_float_abort",
        "local a: Array<float> = [1.0, 2.0, 3.0, 4.0, 5.0]\n"
        "local acc: float = 0.0\n"
        "for s = 1, 20 do\n"
        "  for i = 0, 4 do\n"
        "    acc = acc + a[i]:unwrap()\n"
        "  end\n"
        "end\n"
        "println(acc:to_int())\n",
        300,
        ("regression",),
    )
    yield Case(
        "regression/nested_struct_field_hang",
        "struct Body\n  x: float\nend\n"
        "local b = Body { x = 2.0 }\n"
        "local acc: float = 0.0\n"
        "for s = 1, 20 do\n"
        "  for i = 0, 4 do\n"
        "    acc = acc + b.x\n"
        "  end\n"
        "end\n"
        "println(acc:to_int())\n",
        200,
        ("regression",),
    )
    # 0xFFFFFF80 instead of -128: a 64-bit integer round-tripped through 32 bits.
    yield Case(
        "regression/int_32bit_truncation",
        "struct Counter\n  n: int\nend\n"
        "impl Counter\n"
        "  function bump(self): int\n    self.n = self.n + 1\n    return self.n\n  end\n"
        "end\n"
        "local c = Counter { n = 0 }\n"
        "local acc: int = 0\n"
        "for i = 1, 100 do\n"
        "  acc = acc + c:bump()\n"
        "end\n"
        "println(acc)\n",
        5050,
        ("regression",),
    )
    # Descending numeric for.  The compiler emitted `Le(var, end)` regardless of
    # the step's sign, so this ran zero iterations in *both* modes.
    yield Case(
        "regression/for_negative_step",
        "local acc: int = 0\n"
        "for i = 5, 1, -1 do\n"
        "  acc = acc + 1\n"
        "end\n"
        "println(acc)\n",
        5,
        ("regression", "frontend"),
    )
    # Same, with a step whose sign is only known at runtime: exercises the
    # dynamic direction test rather than the constant-folded one.
    yield Case(
        "regression/for_dynamic_negative_step",
        "local step: int = 0 - 2\n"
        "local acc: int = 0\n"
        "for i = 10, 1, step do\n"
        "  acc = acc + 1\n"
        "end\n"
        "println(acc)\n",
        5,
        ("regression", "frontend"),
    )
    yield Case(
        "regression/for_dynamic_positive_step",
        "local step: int = 3\n"
        "local acc: int = 0\n"
        "for i = 1, 10, step do\n"
        "  acc = acc + 1\n"
        "end\n"
        "println(acc)\n",
        4,
        ("regression", "frontend"),
    )
    # A register holding an inner array was specialized at trace entry and then
    # overwritten by `GetIndex` every iteration; the postamble rebox wrote the
    # entry-time row back over a different row of `grid`.
    yield Case(
        "regression/nested_array_variable_outer_index",
        "local grid: Array<Array<int>> = [[1, 2], [3, 4], [5, 6], [7, 8]]\n"
        "local acc: int = 0\n"
        "local ctr: int = 0\n"
        "for iter_a = 1, 20 do\n"
        "  for iter_b = 1, 5 do\n"
        "    acc = acc + grid[ctr % 4]:unwrap()[ctr % 2]:unwrap()\n"
        "    ctr = ctr + 1\n"
        "  end\n"
        "end\n"
        "println(acc)\n",
        450,
        ("regression",),
    )
    # `jit_unbox_array_int` used to move the buffer out of the source array,
    # leaving it empty for the span of the trace; the nested `for elem in arr`
    # then built its iterator from a zero-length array.
    yield Case(
        "regression/nested_for_in_over_specialized_array",
        "local arr: Array<int> = [1, 2, 3]\n"
        "local acc: int = 0\n"
        "for iter_a = 1, 10 do\n"
        "  for elem in arr do\n"
        "    acc = acc + elem\n"
        "  end\n"
        "end\n"
        "println(acc)\n",
        60,
        ("regression",),
    )
    # A method on a primitive receiver cannot be executed by the compiled trace.
    # The trace used to fail mid-body with the registers already mutated, and the
    # interpreter then re-ran the iteration, double-counting it (57 not 55).
    yield Case(
        "regression/int_method_in_hot_loop",
        "local acc: int = 0\n"
        "for i = 1, 10 do\n"
        "  local neg: int = 0 - i\n"
        "  acc = acc + math.abs(neg)\n"
        "end\n"
        "println(acc)\n",
        55,
        ("regression",),
    )
    # `as` is a non-trapping extraction to Option<T>. The first invocation
    # records the matching path; the same native trace must correctly side-exit
    # when the second invocation receives a different runtime type.
    yield Case(
        "regression/unknown_try_cast_hot_loop",
        "function sum_cast(value: unknown): int\n"
        "  local acc: int = 0\n"
        "  for i = 1, 20 do\n"
        "    if value as int is Some(x) then\n"
        "      acc = acc + x\n"
        "    end\n"
        "  end\n"
        "  return acc\n"
        "end\n"
        "println(sum_cast(3) + sum_cast(\"no\"))\n",
        60,
        ("regression", "unknown"),
    )
    # Plain `is` remains a bool-producing type test and narrows the source in
    # its successful branch; it has its own native trace operation.
    yield Case(
        "regression/unknown_type_is_hot_loop",
        "function sum_is(value: unknown): int\n"
        "  local acc: int = 0\n"
        "  for i = 1, 20 do\n"
        "    if value is int then\n"
        "      acc = acc + value\n"
        "    end\n"
        "  end\n"
        "  return acc\n"
        "end\n"
        "println(sum_is(3) + sum_is(\"no\"))\n",
        60,
        ("regression", "unknown"),
    )
    yield Case(
        "regression/array_index_result",
        "local values: Array<int> = [10, 20]\n"
        "local acc: int = 0\n"
        "if values[1] is Ok(value) then\n"
        "  acc = acc + value\n"
        "end\n"
        "if values[2] is Err(error) then\n"
        "  acc = acc + error.index * 10 + error.length\n"
        "end\n"
        "println(acc)\n",
        42,
        ("regression", "array", "result"),
    )
    yield Case(
        "regression/array_index_ok_hot_loop",
        "local values: Array<int> = [1, 2, 3, 4]\n"
        "local acc: int = 0\n"
        "for i = 0, 99 do\n"
        "  if values[i % 4] is Ok(value) then\n"
        "    acc = acc + value\n"
        "  end\n"
        "end\n"
        "println(acc)\n",
        250,
        ("regression", "array", "result", "hot"),
    )
    yield Case(
        "regression/array_index_mixed_bounds_hot_loop",
        "local values: Array<int> = [1, 2, 3, 4]\n"
        "local acc: int = 0\n"
        "for i = 0, 99 do\n"
        "  if values[i % 5] is Ok(value) then\n"
        "    acc = acc + value\n"
        "  end\n"
        "end\n"
        "println(acc)\n",
        200,
        ("regression", "array", "result", "hot"),
    )
    many_values = ", ".join(["10"] * 101)
    yield Case(
        "regression/array_index_alias_after_mutation",
        f"local values: Array<int> = [{many_values}]\n"
        "local alias = values\n"
        "local acc: int = 0\n"
        "for i = 1, 100 do\n"
        "  array.pop(alias)\n"
        "  if alias[0] is Ok(value) then\n"
        "    acc = acc + value\n"
        "  end\n"
        "end\n"
        "println(acc)\n",
        1000,
        ("regression", "array", "alias", "result", "hot"),
    )
    yield Case(
        "regression/array_unwrap_alias_after_mutation",
        f"local values: Array<int> = [{many_values}]\n"
        "local alias = values\n"
        "local acc: int = 0\n"
        "for i = 1, 100 do\n"
        "  array.pop(alias)\n"
        "  acc = acc + alias[0]:unwrap()\n"
        "end\n"
        "println(acc)\n",
        1000,
        ("regression", "array", "alias", "unwrap", "hot"),
    )
    yield Case(
        "regression/failed_unbox_with_later_specialization",
        "local mixed: Array<unknown> = [1, \"not an int\"]\n"
        "local values: Array<int> = [1, 2, 3, 4, 5, 6, 7, 8]\n"
        "local acc: int = 0\n"
        "for i = 1, 20 do\n"
        "  acc = acc + array.len(values)\n"
        "end\n"
        "println(acc)\n",
        160,
        ("regression", "array", "specialization", "hot"),
    )
    # A type parameter is generic only inside the declaration that introduces it;
    # a one-letter PascalCase name remains a normal nominal type everywhere else.
    yield Case(
        "regression/single_letter_struct_name",
        "struct K\n  x: int\nend\n"
        "local a: Array<K> = []\n"
        "array.push(a, K { x = 7 })\n"
        "println(a[0]:unwrap().x)\n",
        7,
        ("regression", "generics", "scope"),
    )
    yield Case(
        "regression/generic_function_inference",
        "function identity<Element>(value: Element): Element\n"
        "  return value\n"
        "end\n"
        "function first<Element>(values: Array<Element>): Element\n"
        "  return values[0]:unwrap()\n"
        "end\n"
        "function tagged<Tag>(value: int): int\n"
        "  return value\n"
        "end\n"
        "println(identity(7) + first([5]) + tagged<string>(2))\n",
        14,
        ("regression", "generics", "function", "nested"),
    )
    yield Case(
        "regression/generic_struct_impl",
        "struct Box<Item>\n"
        "  value: Item\n"
        "end\n"
        "impl<Item> Box<Item>\n"
        "  function get(self): Item\n"
        "    return self.value\n"
        "  end\n"
        "  function replace<Next>(self, value: Next): Next\n"
        "    return value\n"
        "  end\n"
        "end\n"
        "local boxed: Box<int> = Box { value = 7 }\n"
        "println(boxed:get() + boxed:replace(3))\n",
        10,
        ("regression", "generics", "struct", "method"),
    )
    yield Case(
        "regression/generic_trait_bound_and_value",
        "trait Scaled\n"
        "  function scale(self, value: int): int\n"
        "end\n"
        "struct Multiplier\n"
        "  factor: int\n"
        "end\n"
        "impl Scaled for Multiplier\n"
        "  function scale(self, value: int): int\n"
        "    return self.factor * value\n"
        "  end\n"
        "end\n"
        "function apply<Item: Scaled>(item: Item, value: int): int\n"
        "  return item:scale(value)\n"
        "end\n"
        "function apply_dynamic(item: Scaled, value: int): int\n"
        "  return item:scale(value)\n"
        "end\n"
        "local multiplier = Multiplier { factor = 3 }\n"
        "println(apply(multiplier, 7) + apply_dynamic(multiplier, 5))\n",
        36,
        ("regression", "generics", "trait", "dynamic"),
    )
    yield Case(
        "regression/generic_enum_pattern_and_method",
        "enum Boxed<Item>\n"
        "  Value(Item)\n"
        "end\n"
        "impl<Item> Boxed<Item>\n"
        "  function constant(self): int\n"
        "    return 4\n"
        "  end\n"
        "end\n"
        "local boxed: Boxed<int> = Boxed.Value(9)\n"
        "if boxed is Value(value) then\n"
        "  println(value + boxed:constant())\n"
        "else\n"
        "  println(0)\n"
        "end\n",
        13,
        ("regression", "generics", "enum", "pattern", "method"),
    )
    yield Case(
        "regression/builtin_trait_value",
        "struct Label\n"
        "  value: string\n"
        "end\n"
        "impl ToString for Label\n"
        "  function to_string(self): string\n"
        "    return self.value\n"
        "  end\n"
        "end\n"
        "function size(value: ToString): int\n"
        "  return string.len(value:to_string())\n"
        "end\n"
        "local label = Label { value = \"four\" }\n"
        "local dynamic: unknown = label\n"
        "if dynamic is ToString then\n"
        "  println(size(label))\n"
        "else\n"
        "  println(0)\n"
        "end\n",
        4,
        ("regression", "trait", "builtin", "dynamic"),
    )
    yield Case(
        "regression/trait_cast_value",
        "trait Scaled\n"
        "  function scale(self, value: int): int\n"
        "end\n"
        "struct Multiplier\n"
        "  factor: int\n"
        "end\n"
        "impl Scaled for Multiplier\n"
        "  function scale(self, value: int): int\n"
        "    return self.factor * value\n"
        "  end\n"
        "end\n"
        "local dynamic: unknown = Multiplier { factor = 3 }\n"
        "local scaled: Scaled = (dynamic as Scaled):unwrap()\n"
        "println(scaled:scale(7))\n",
        21,
        ("regression", "trait", "cast", "dynamic"),
    )


def all_cases() -> Iterator[Case]:
    seen: set[str] = set()
    for gen in (regressions, loop_shapes, body_ops, hot_sweep):
        for case in gen():
            if case.name in seen:
                raise AssertionError(f"duplicate case name: {case.name}")
            seen.add(case.name)
            yield case
