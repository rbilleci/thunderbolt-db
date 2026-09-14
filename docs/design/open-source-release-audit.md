# Open-source release audit — 2026-09-14

This document is the engineering and license-risk evidence for the completed **OSS-001** milestone. It is not
legal advice and does not authorize publication. Public push, repository visibility, and release publication
remain owner actions.

The baseline audited revision is `6508ac9bafcf6cba04506fc77846b0ecdae31416` (2026-08-10). The release
candidate is an **experimental, single-node, GPU-required source release**. It does not claim production readiness,
full PostgreSQL compatibility, high availability, scale completion, or a comparative performance result.

## Candidate status

The integrated runtime candidate, commit `139aa8959905877c3fac6809d0f5568b1fd33c1c` and tree
`5c95d112891cbd9bb69c562600efcd5aa563718f`, addresses every technical finding from the baseline audit. On
2026-09-14, the release authority confirmed that they own or are authorized to license the project-authored
material under GPL-3.0-only with the CUDA Driver Additional Permission and are unaware of conflicting employer,
client, school, or contributor rights. They selected the public contributor-form copyright notice rather than
publishing a legal personal name. The documentation-only closeout commit and annotated local `v0.1.0-alpha.1` tag
preserve this exact runtime tree; the tag message records the final source archive SHA-256 and size.

| OSS-001 row | Final disposition | State |
|---|---|---|
| Rights and license | Full GPLv3 text; approved `GPL-3.0-only` contributor-form project notice; narrow section-7 CUDA Driver Additional Permission; third-party notices and license texts; consistent metadata across 17 crates. | Accepted. |
| Dependency hygiene | Vulnerable development clients and yanked transitive package upgraded; both unmaintained dependencies removed; locked deny policy passes without advisory exceptions; inventory regenerated. | Accepted. |
| Publication contents | Gitleaks all-history and exact-tree scans reviewed; source archive excludes only the separately unreviewed literature corpus; Rust/CUDA/PTX/header/build/test inputs remain. | Accepted; repository history is outside the prepared source release. |
| Working route | README gives the supported platform, native prerequisites, locked build, durable WAL, loopback trust profile, psql SQL/restart workflow, security profile, and limits. A 10-assertion GPU smoke automates it. | Accepted on the audit GPU and in a clean Ubuntu 26.04 container. |
| Exact release proof | Hosted and trusted-GPU workflows, pinned Rust, source builder, focused recovery/pgwire/security gates, non-vacuous GPU checks, canonical performance comparison, deterministic archive, and independent audits are present. | Accepted for the local experimental source release. |

## Rights, licensing, and CUDA boundary

The baseline had no tracked license text and all workspace packages declared MIT. The candidate adds:

- the unmodified GNU GPL version 3 text in `LICENSE`;
- `GPL-3.0-only` workspace metadata for all 17 project crates;
- a project notice in `COPYRIGHT`;
- `CUDA_EXCEPTION`, a narrow GPLv3 section-7 permission covering only unmodified, separately obtained NVIDIA
  components that implement the published CUDA Driver API;
- `THIRD_PARTY_NOTICES.md` and the reviewed license texts under `THIRD_PARTY_LICENSES/`;
- DCO 1.1 contribution sign-off and provenance requirements.

The engine dynamically loads `libcuda.so.1` and CUDA Driver API `cu*` symbols. The driver is essential at
runtime, and no NVIDIA binary, CUDA Toolkit library, SDK header, or sample source is distributed. The additional
permission avoids relying only on a fact-sensitive System Library interpretation. It excludes the CUDA Runtime,
Toolkit libraries and tools, static NVIDIA libraries, NVRTC, nvJitLink, cuBLAS, cuDNN, NCCL, and NVML.

The frozen runtime candidate has 3,701 commits in its reachable history. Git author records use three apparent variants of one
contributor identity; 1,152 historical commits also contain automated-assistant co-author trailers. Neither an
author record nor a trailer proves copyright ownership, employer/client clearance, or the right to grant an
exception. The release authority's 2026-09-14 declaration covers all project-authored Rust, CUDA, PTX, headers,
documentation, scripts, tests, and media in the source artifact, including material created with automated tools.
The public notice names the project contributors; the release audit does not publish a legal personal identity.

If any version was validly distributed under MIT, the GPL release does not retract that prior grant. The candidate
licenses the new source release under GPL-3.0-only with the stated additional permission.

Relevant primary terms:
[GPLv3](https://www.gnu.org/licenses/gpl-3.0.html) and
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html).

## Dependency and notice evidence

The [resolved inventory](open-source-dependencies.tsv) contains 313 external packages from
`cargo metadata --locked --offline --format-version 1`, including development, build, and target-specific
packages. It is an exact component/license inventory for the locked Cargo graph, not an SPDX SBOM.

The candidate changes the relevant packages as follows:

| Baseline finding | Candidate disposition |
|---|---|
| `postgres-protocol 0.6.11`, RUSTSEC-2026-0179/0180 | Upgraded to 0.6.12. |
| `tokio-postgres 0.7.17`, RUSTSEC-2026-0178 | Upgraded to 0.7.18. |
| Yanked `chacha20 0.10.0` | Updated to 0.10.2 through the refreshed client graph. |
| Unmaintained `bitmaps 3.2.1` | Removed by upgrading to `imbl 7.0.2` / `imbl-sized-chunks 0.2.0`. |
| Unmaintained `rustls-pemfile 2.2.0` | Removed; server and replication PEM loading use rustls `pki_types::pem::PemObject`. |

`cargo-deny 0.19.9 --locked check advisories bans licenses sources` passes with no advisory, license, or source
exceptions. Nine informational duplicate-version warnings remain in transitive build/development graphs
(`getrandom`, `itertools`, `linux-raw-sys`, `r-efi`, `rand_core`, `rustix`, `shlex`,
`windows-sys`, and `wit-bindgen`). They represent upstream version splits, are visible under the policy's
`multiple-versions = "warn"`, and are not undisclosed release findings.

Special native/component review is recorded for:

- `pg_query 6.1.1`, which builds libpg_query, PostgreSQL 17.4-derived parser source, protobuf-c, and xxHash;
- `aws-lc-rs 1.17.0` / `aws-lc-sys 0.41.0`;
- `tikv-jemallocator 0.6.1` / jemalloc;
- `imbl 7.0.2` and `imbl-sized-chunks 0.2.0` under MPL-2.0 or later, with no inspected Exhibit B
  incompatibility notice.

The source artifact does not vendor Cargo dependency sources. Anyone distributing binaries must re-evaluate the
exact target graph and carry every applicable upstream notice.

## Publication scope and provenance

The prepared artifact is a deterministic `git archive` snapshot compressed with `gzip -n`. It has no Git
history. `scripts/build_source_release.sh` binds the archive to one committed ref and one workspace version,
requires every project and third-party license file, rejects excluded/build paths, and prints its SHA-256 and size.

The source snapshot retains all 27 checked-in PTX files, three preferred-form `.cu` files, the project
`.cuh` header, adjacent PTX regeneration commands, Rust sources, manifests, lockfile, tests, examples, scripts,
and contributor/acceptance configuration. Three PTX artifacts record generation by CUDA 12.4 / V12.4.131; the
remaining PTX files are handwritten project sources. No tracked CUDA SDK header or NVIDIA sample source was found.

The literature-review corpus was excluded because its papers, extracted text, generated JSON, and image provenance
is independent of program source and did not receive publication clearance. It is not a build, test, modification,
or runtime input. The initial `v0.1.0-alpha.1` tree retained other historical project evidence for then-live links;
the subsequent archive cleanup removed that material from the current tree while preserving it in Git history.

Gitleaks 8.28.0 scanned all refs/history with `--log-opts="--all"`. Three generic-key findings were reviewed as
prose false positives and are narrowly fingerprinted in `.gitleaksignore`; one stable exact text pattern is
allowlisted in `.gitleaks.toml`. The repeated all-history scan and a separate exact current-source scan both
returned no leak. No credential value is reproduced in this audit.

The GitHub repository was read-only verified as private on 2026-09-14. Making it public would publish the Git
history, including the excluded research corpus. That is outside this source-release candidate and requires a
separate scope/provenance decision even though the secret scan is clean.

## Working solution and portability evidence

The audit host is Ubuntu 26.04 with Rust/Cargo 1.97.1, an RTX PRO 6000 Blackwell Max-Q GPU, and NVIDIA driver
595.84. This is a tested configuration, not a memory or broader-hardware claim. The supported floor remains
Blackwell / compute capability 12.0.

| Check | Result |
|---|---|
| `scripts/check_product_ownership.sh` | Passed: one product server and canonical commit/WAL/publication ownership. |
| `durable_process_recovery` | 1/1 passed; real process SIGKILL/restart, typed prepared writes, NULLs, COPY, DDL/DML, rollback, and continued appends. |
| `pgwire_roundtrip --include-ignored` | 6/6 passed; asynchronous/blocking clients, COPY/recovery/hazard, concurrent connections, and a non-vacuous GPU point route. |
| `scripts/run_oss_release_smoke.sh` using the rebuilt release binary | 10/10 assertions passed: typed/NULL insert, committed update, rolled-back delete, ordered GPU read and SUM, SIGKILL, restart, and identical recovered results. |
| Twice-built frozen runtime source archive | Byte-identical outputs, SHA-256 `de8ff5522eba7f874fc18930ffa335314158ec8439e31c5802aa798dc579746c`, 5,931,826 bytes; 27 PTX / 3 CUDA / 1 header; zero research or target entries. This hash binds runtime commit `139aa895`; the annotated local tag records the final documentation-only closeout archive. |
| Extracted runtime-candidate archive in a new Ubuntu 26.04 container | Clean locked release build passed in 3m39s with Rust 1.97.1, Clang/libclang 21.1.8, PostgreSQL client 18.6, the documented native packages, and no bindgen workaround. Product ownership and all 10 durable GPU SQL/SIGKILL/restart assertions passed against the container-built binary with the host RTX PRO 6000/driver 595.84 passed through. |
| Connection security preflight | Passed with generated TLS certificate/key PEM and SCRAM authentication. |
| Replication channel security preflight | Passed with generated mTLS CA/node PEM and an authenticated append route. |
| `cargo check --locked --workspace --all-targets --all-features` | Passed with the disclosed audit-host bindgen include workaround. |
| `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings` | Passed after six borrow-only lint corrections in the typed insert generation path. |
| Host-neutral locked workspace test subset | Passed across all CPU-safe crates with the four GPU/product crates excluded as in hosted CI; device-only tests were ignored rather than counted as GPU evidence. |
| `cargo fmt --check`, `git diff --check`, shell syntax, and workflow YAML parse | Passed before freeze. |

The former `.cargo/config.toml` embedded a GCC-15 include path and a repository-local TMPDIR. It is removed.
The README now names the actual Ubuntu native packages, pins Rust 1.97.1 through `rust-toolchain.toml`, and makes
TMPDIR explicit. This audit host lacks the documented `clang` and `libclang-dev` packages and cannot install them
without interactive sudo; an unprovisioned host build therefore reproduces bindgen's missing `stddef.h` error.
The pre-freeze release build passes there with the disclosed temporary GCC-15 include workaround. More
significantly, the extracted artifact builds without any workaround in a clean Ubuntu 26.04 container after
installing the documented packages. The first minimal container run also showed that release verification scripts
need Python 3 and the ownership guard needs ripgrep; both are now explicit prerequisites. The exact tagged closeout
archive passed the same clean-container gate; its archive and transcript identities are stored in the annotated tag
outside the self-referential source payload.

The hosted workflow installs native prerequisites, pins Rust and policy-tool versions, checks the complete locked
workspace, and runs host-neutral tests. GPU absence is reported there. The manual trusted workflow instead fails
on missing NVIDIA hardware, bounds every gate, runs ownership, recovery, the complete pgwire matrix, all ignored
GPU tests, and the durable source-release smoke, and verifies a positive executed-test count.

The `imbl` upgrade affects production MVCC storage and engine value indexes, so the full report card was applicable.
The independent pre-card audit accepted the frozen runtime candidate before measurement. Both canonical cards used
Rust/Cargo 1.97.1, `BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/gcc/x86_64-linux-gnu/15/include`, the fixed full workload,
fresh targets, the same RTX PRO 6000 Blackwell Max-Q / driver 595.84 identity, the same configuration hash, and an
exclusive GPU. Each has exactly one terminal canonical record, five complete section markers, and seven valid
environment samples with zero external contexts.

The direct-parent base is `6508ac9bafcf6cba04506fc77846b0ecdae31416`, tree
`19a93c11e73fe02a1299b84b2e8a005e39f0afbd`. Its transcript is
`runner.6508ac9bafcf.19a93c11e73f.dLhxaG.log`, SHA-256
`9a4ea081124b2f1f5c3b8411386d811f71ea87d4d8af07e209bdfe5977f53a52`. The candidate transcript is
`runner.139aa8959905.5c95d112891c.quAEjj.log`, SHA-256
`b384c05132baec7b7d7cfcd52d3fe4a03f7aeed3851428c0c90d4f11880dd268`.

At batch 65,536, the base/candidate in-L2 medians were 263,417,009 / 262,173,070 lookups/s (-0.473%); all three
candidate samples exceeded the 260M floor. Out-of-L2 results were 256,048,869 / 258,583,015 lookups/s (+0.990%)
at the same 126us p50, and fixture construction was 225.0 / 228.7 seconds (+1.64%). Raw ratios were stable:
in-L2 constant-mask/roofline moved from 0.824 to 0.814, out-of-L2 from 1.038 to 1.037, grouped execution was flat,
and the host-H2D-inclusive hash join moved from 239.7 to 246.7 M-elem/s. The independent post-card audit returned
**ACCEPT** with no material regression.

The first candidate invocation, transcript SHA-256
`99a43f49d8da9d8be106977ca69e553b8c2b4a11c5f19d2fdd8a1557d97d8573`, failed before Section A because the audit
host lacks the documented Clang resource headers. It has no section marker or terminal status and is retained only
as incomplete environmental evidence. A completed card at `b24889c` is also non-acceptance historical evidence:
it is not an ancestor of the candidate and cannot isolate the OSS changes. The candidate card was not rerun after
the valid direct-parent comparison. No comparative performance claim is made by this release.

## Release boundary

The accepted deliverable is the deterministic source archive at the annotated local `v0.1.0-alpha.1` tag. Its tag
message is the checksum record for the exact documentation-closeout archive. It excludes Git history and the
uncleared literature corpus. The current tree subsequently removes all other historical archives; the accepted tag
remains the immutable original release boundary. Public push, a visibility change, binary/container distribution,
and release publication were not performed.
