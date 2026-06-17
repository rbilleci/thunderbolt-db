#!/usr/bin/env python3
"""P0 §9.6 tight-widening driver for impl-Engine split slices.

After moving a method group into a new `impl Engine` module, the only callers
that break are the *moved* methods reached from lib.rs / tests / sibling modules
(E0624 private-method). Methods left in lib.rs keep descendant access, so every
flagged method lives in the just-created module file. This loops `cargo check
--all-targets`, collects the E0624 method names, and widens exactly those (and no
others) to pub(crate) in the given module file, until the build is clean.

Usage: decomp_widen.py crates/engine/src/<module>.rs
"""
import re
import subprocess
import sys

mod = sys.argv[1]


def check():
    p = subprocess.run(["cargo", "check", "-p", "gpu_db_engine", "--all-targets",
                        "--message-format=short"], capture_output=True, text=True)
    return p.stderr


def flagged(out):
    names = set()
    for m in re.finditer(r"(?:method|associated function) `([a-z_0-9]+)` is private", out):
        names.add(m.group(1))
    return names


total = 0
for _ in range(20):
    out = check()
    names = flagged(out)
    if not names:
        break
    with open(mod) as f:
        lines = f.readlines()
    n = 0
    for i, ln in enumerate(lines):
        m = re.match(r"^    (async )?fn ([a-z_0-9]+)", ln)
        if m and m.group(2) in names:
            lines[i] = "    pub(crate) " + ln[4:]
            n += 1
            names.discard(m.group(2))
    with open(mod, "w") as f:
        f.writelines(lines)
    total += n
    if n == 0:
        sys.stderr.write(f"NO PROGRESS; unresolved: {names}\n")
        sys.stderr.write(out[-2000:])
        sys.exit(1)

out = check()
remaining = [l for l in out.splitlines() if l.startswith("error")]
print(f"widened {total} methods in {mod}; remaining errors: {len(remaining)}")
for l in remaining[:15]:
    print(l)
