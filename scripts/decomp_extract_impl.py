#!/usr/bin/env python3
"""P0 §9.6 impl-Engine method-group extractor (behavior-preserving).

Move a cohesive run of `impl Engine` methods into a new sibling module that
re-opens `impl Engine { ... }`. You give the *fn-line* of the FIRST and LAST
method in the group (from the method map); the tool auto-aligns to whole methods:
it backs the start up over the first method's leading doc-comments/attributes,
and brace-finds the last method's close (rustfmt puts a method's closing brace at
the unique 4-space-indented `    }`). Methods keep their visibility; cross-module
callers are fixed up afterwards by the compiler (tight pub(crate) widening of only
the entry methods actually reached from lib.rs / sibling modules). Moved methods
keep descendant access to Engine's crate-root private fields and to methods still
in lib.rs, so only the *moved* methods called from outside need widening.

Usage: decomp_extract_impl.py <first_fn_line> <last_fn_line> <module> <<<header
"""
import re
import sys

LIB = "crates/engine/src/lib.rs"


def main():
    first_fn, last_fn, module = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    header = sys.stdin.read().rstrip("\n")
    with open(LIB) as f:
        lines = f.readlines()

    # start: back up over the first method's leading doc-comments / attributes
    start = first_fn
    while start - 1 >= 1:
        prev = lines[start - 2].rstrip()
        if re.match(r"^    (///|//!|//|#\[)", prev) or (prev.endswith("]") and prev.lstrip().startswith("#")):
            start -= 1
        else:
            break
    assert re.match(r"^    (pub |pub\(crate\) )?(async )?fn ", lines[first_fn - 1]), \
        f"first_fn line is not a method: {lines[first_fn-1]!r}"

    # end: from the last method's fn line, find its closing `    }` (4-space indent)
    end = None
    for i in range(last_fn, len(lines) + 1):
        if lines[i - 1].rstrip() == "    }":
            end = i
            break
    assert end is not None, "could not find method close for last_fn"

    block = lines[start - 1:end]
    body = "".join(block).strip("\n")
    with open(f"crates/engine/src/{module}.rs", "w") as f:
        f.write(f"{header}\n\nuse super::*;\n\nimpl Engine {{\n{body}\n}}\n")
    # anchor on the top re-export cluster only (`pub use`/`pub(crate) use` are
    # top-of-file); plain `mod ...;` is NOT matched because `mod tests;` lives at
    # the very end and would push the insertion past the cut range.
    ins = max(i for i, l in enumerate(lines)
              if re.match(r"^(pub use |pub\(crate\) use )", l))
    new_lib = lines[:ins + 1] + [f"mod {module};\n"] + lines[ins + 1:start - 1] + lines[end:]
    with open(LIB, "w") as f:
        f.writelines(new_lib)
    print(f"{module}: moved lines {start}-{end} ({end-start+1} lines) into a fresh impl Engine block; "
          f"lib.rs {len(lines)} -> {len(new_lib)}")


if __name__ == "__main__":
    main()
