Regression programs for the tracing JIT, run by `examples/jit_diff.py` with
the JIT on and off. Each one reproduces a way a trace once diverged from the
interpreter: exits from inlined callees, guards on values that change
mid-loop, nested loops, specialized arrays across guard exits.
