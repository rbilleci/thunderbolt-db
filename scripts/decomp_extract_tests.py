#!/usr/bin/env python3
"""P0 §9.6 test-suite splitter: move a contiguous run of tests (+ their local
fixtures/helpers) out of crates/engine/src/tests/mod.rs into a feature submodule
crates/engine/src/tests/<feature>.rs.

You give the line of the FIRST item (test/struct/helper) and the FN line of the
LAST test in the run; the tool back-aligns the start over the first item's
attrs/doc-comments and brace-finds the last item's col-0 close. Header defaults to
`use super::*;` (super = crate::tests = mod.rs, which itself `use super::*` the
crate root) plus `use crate::*;` for the engine items directly — override via
$HEADER. Adds `mod <feature>;` after mod.rs's `use super::*;`.

Usage: decomp_extract_tests.py <first_item_line> <last_fn_line> <feature>
       HEADER="use super::*;" decomp_extract_tests.py ...
"""
import os
import re
import sys

MOD = "crates/engine/src/tests/mod.rs"
HEADER = os.environ.get("HEADER", "use super::*;")


def main():
    first, last_fn, feature = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    with open(MOD) as f:
        lines = f.readlines()

    # back-align start over the first item's leading col-0 attrs / doc comments
    start = first
    while start - 1 >= 1:
        prev = lines[start - 2].rstrip()
        if re.match(r"^(#\[|///|//!|//)", prev) or (prev.startswith("#") and prev.endswith("]")):
            start -= 1
        else:
            break

    # brace-find the last test's close: first col-0 `}` at/after last_fn
    end = None
    for i in range(last_fn, len(lines) + 1):
        if lines[i - 1].rstrip() == "}":
            end = i
            break
    assert end is not None, "no col-0 close found for last_fn"

    block = lines[start - 1:end]
    with open(f"crates/engine/src/tests/{feature}.rs", "w") as f:
        f.write(HEADER.rstrip("\n") + "\n\n" + "".join(block).strip("\n") + "\n")

    # insert `mod <feature>;` right after the first `use super::*;`
    ins = next(i for i, l in enumerate(lines) if l.rstrip() == "use super::*;")
    new = lines[:ins + 1] + [f"mod {feature};\n"] + lines[ins + 1:start - 1] + lines[end:]
    with open(MOD, "w") as f:
        f.writelines(new)
    print(f"{feature}: moved lines {start}-{end} ({end-start+1} lines); mod.rs {len(lines)} -> {len(new)}")


if __name__ == "__main__":
    main()
