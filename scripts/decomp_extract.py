#!/usr/bin/env python3
"""P0 §9.6 behavior-preserving slice extractor.

Move a contiguous block [start,end] (1-based inclusive) of lib.rs into a new
sibling module, widening the items the rest of the crate reaches (free fns,
private structs/enums/consts/type-aliases, inherent-impl methods, and struct
fields) to pub(crate) so a glob re-export keeps every call site unchanged.
Public (`pub`) items keep their visibility. Trait-impl methods and enum-variant
fields are left alone (they can't take a visibility modifier). Struct fields are
widened only inside the body (after the opening `{`), so generic where-clauses
are not mistaken for fields.

Usage: decomp_extract.py <start> <end> <module> <reexport:pub|pubcrate> <<<header
"""
import re
import sys

LIB = "crates/engine/src/lib.rs"


def transform(block):
    state = None  # struct_hdr|struct_body|enum|impl_inherent|impl_trait|fn|None
    field_re = re.compile(r"^    ([A-Za-z_]\w*): ")
    out, n = [], {"fn": 0, "type": 0, "field": 0, "method": 0}
    for ln in block:
        if re.match(r"^fn ", ln):
            out.append("pub(crate) " + ln); n["fn"] += 1; state = "fn"; continue
        if re.match(r"^(const|type|static) ", ln):
            out.append("pub(crate) " + ln); n["type"] += 1; state = None; continue
        m = re.match(r"^(pub(\(crate\))? )?struct ", ln)
        if m:
            if not m.group(1):
                ln = "pub(crate) " + ln
            state = "struct_body" if ln.rstrip().endswith("{") else "struct_hdr"
            out.append(ln); continue
        m = re.match(r"^(pub(\(crate\))? )?enum ", ln)
        if m:
            if not m.group(1):
                ln = "pub(crate) " + ln
            state = "enum"; out.append(ln); continue
        if ln.startswith("impl "):
            state = "impl_trait" if " for " in ln else "impl_inherent"
            out.append(ln); continue
        if ln.startswith("}"):
            state = None; out.append(ln); continue
        if state == "struct_hdr" and ln.rstrip().endswith("{"):
            state = "struct_body"; out.append(ln); continue
        if state == "struct_body" and field_re.match(ln) and not ln.lstrip().startswith("pub"):
            out.append("    pub(crate) " + ln[4:]); n["field"] += 1; continue
        if state == "impl_inherent" and re.match(r"^    fn ", ln):
            out.append("    pub(crate) " + ln[4:]); n["method"] += 1; continue
        out.append(ln)
    return out, n


def main():
    start, end, module, reexport = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    header = sys.stdin.read()
    with open(LIB) as f:
        lines = f.readlines()
    block = lines[start - 1:end]
    assert block[-1].strip() == "", f"block must end on a blank line, got {block[-1]!r}"
    out, n = transform(block)
    with open(f"crates/engine/src/{module}.rs", "w") as f:
        f.write(header.rstrip("\n") + "\n\nuse super::*;\n\n" + "".join(out).rstrip("\n") + "\n")
    vis = "pub" if reexport == "pub" else "pub(crate)"
    # insert after the last existing module re-export near the top
    ins = max(i for i, l in enumerate(lines)
              if l.startswith("pub use ") or l.startswith("pub(crate) use "))
    mod_block = [f"mod {module};\n", f"{vis} use {module}::*;\n"]
    new_lib = lines[:ins + 1] + mod_block + lines[ins + 1:start - 1] + lines[end:]
    with open(LIB, "w") as f:
        f.writelines(new_lib)
    print(f"{module}: moved {len(block)} lines "
          f"({n['fn']} fns, {n['type']} const/type, {n['field']} fields, {n['method']} methods -> pub(crate)); "
          f"lib.rs {len(lines)} -> {len(new_lib)}")


if __name__ == "__main__":
    main()
