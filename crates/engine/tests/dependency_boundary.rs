//! Dependency-direction guard for the `engine → protocol` inversion (roadmap §9.2).
//!
//! The neutral SQL vocabulary now lives in `gpu_db_sql`; the engine consumes it
//! from there and must NOT depend on `gpu_db_protocol` (the pgwire/wire crate)
//! again — not directly, not transitively through another workspace crate, and
//! not even as a dev-dependency (a dev-edge would re-introduce protocol into
//! `cargo tree -p gpu_db_engine`, the acceptance bar). This test re-derives the
//! engine's workspace path-dependency closure from the `Cargo.toml` manifests and
//! fails if `gpu_db_protocol` reappears, so the back-edge cannot silently return.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The crate whose dependency closure we forbid here.
const FORBIDDEN: &str = "gpu_db_protocol";

/// `crates/` directory, derived from this test crate's manifest dir.
fn crates_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = .../crates/engine
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("engine crate lives under crates/")
        .to_path_buf()
}

/// Names of the workspace path-dependencies declared in `[dependencies]` (and,
/// when `include_dev` is set, `[dev-dependencies]`) of the given crate's manifest.
/// We only track in-workspace `gpu_db_*` crates; external crates can never reach
/// `gpu_db_protocol` and are irrelevant to the back-edge.
fn workspace_deps(manifest: &Path, include_dev: bool) -> Vec<String> {
    let text = fs::read_to_string(manifest)
        .unwrap_or_else(|err| panic!("read {}: {err}", manifest.display()));

    let mut deps = Vec::new();
    let mut section: Option<&str> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            section = Some(line);
            continue;
        }
        let in_deps = matches!(section, Some(s) if s == "[dependencies]")
            || (include_dev && matches!(section, Some(s) if s == "[dev-dependencies]"));
        if !in_deps {
            continue;
        }
        // Dependency lines look like `gpu_db_xyz = { path = "../xyz" }`.
        let Some((name, _)) = line.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name.starts_with("gpu_db_") {
            deps.push(name.to_string());
        }
    }
    deps
}

fn manifest_for(crate_name: &str) -> PathBuf {
    // Workspace convention: crate `gpu_db_foo` lives in `crates/foo`.
    let dir = crate_name.strip_prefix("gpu_db_").unwrap_or(crate_name);
    crates_dir().join(dir).join("Cargo.toml")
}

#[test]
fn engine_does_not_depend_on_protocol_transitively() {
    let start = manifest_for("gpu_db_engine");
    assert!(
        start.exists(),
        "engine manifest missing: {}",
        start.display()
    );

    // BFS over the engine's *normal* (non-dev) workspace dependency closure.
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut stack: Vec<String> = workspace_deps(&start, false);
    let mut path_into_forbidden: Option<String> = None;

    while let Some(dep) = stack.pop() {
        if dep == FORBIDDEN {
            path_into_forbidden = Some(dep);
            break;
        }
        if !seen.insert(dep.clone()) {
            continue;
        }
        let manifest = manifest_for(&dep);
        if manifest.exists() {
            stack.extend(workspace_deps(&manifest, false));
        }
    }

    assert!(
        path_into_forbidden.is_none(),
        "engine→{FORBIDDEN} back-edge returned: {FORBIDDEN} is reachable through the \
         engine's normal dependency closure. The neutral SQL vocabulary must come from \
         gpu_db_sql; route the parser/AST through that crate, not the wire crate \
         (roadmap §9.2)."
    );
}

#[test]
fn engine_does_not_dev_depend_on_protocol() {
    // A dev-dependency on the wire crate would still show up in
    // `cargo tree -p gpu_db_engine` (the acceptance bar), so forbid it too. The
    // engine+pgwire integration examples live in the `server` crate, which is the
    // layer that legitimately spans the engine and the wire crate.
    let manifest = manifest_for("gpu_db_engine");
    let dev = workspace_deps(&manifest, true);
    assert!(
        !dev.iter().any(|d| d == FORBIDDEN),
        "engine declares a dev-dependency on {FORBIDDEN}; that re-introduces it into \
         `cargo tree -p gpu_db_engine`. Keep wire-integration examples in the server crate."
    );
}
