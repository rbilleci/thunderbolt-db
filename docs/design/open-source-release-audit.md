# Open-source release audit — 2026-09-14

This document is the engineering and license-risk evidence for
**[OSS-001](../PLAN.md#oss-001--minimal-gplv3-source-release)**. It is not legal advice and does not authorize
publication. Public push, repository visibility, and release publication remain owner actions.

The baseline audited revision is `6508ac9bafcf6cba04506fc77846b0ecdae31416` (2026-08-10). The release
candidate is an **experimental, single-node, GPU-required source release**. It does not claim production readiness,
full PostgreSQL compatibility, high availability, scale completion, or a comparative performance result.

## Candidate status

The integrated candidate addresses every technical finding from the baseline audit. On 2026-09-14, the release
authority confirmed that they own or are authorized to license the project-authored material under GPL-3.0-only
with the CUDA Driver Additional Permission and are unaware of conflicting employer, client, school, or contributor
rights. They selected the public contributor-form copyright notice rather than publishing a legal personal name.
The exact commit, source archive hash, local tag, and independent acceptance decision are recorded after freeze.

| OSS-001 row | Candidate disposition | State before freeze |
|---|---|---|
| Rights and license | Full GPLv3 text; approved `GPL-3.0-only` contributor-form project notice; narrow section-7 CUDA Driver Additional Permission; third-party notices and license texts; consistent metadata across 17 crates. | Rights declaration recorded; freeze pending. |
| Dependency hygiene | Vulnerable development clients and yanked transitive package upgraded; both unmaintained dependencies removed; locked deny policy passes without advisory exceptions; inventory regenerated. | Technical checks pass. |
| Publication contents | Gitleaks all-history and exact-tree scans reviewed; source archive excludes only the separately unreviewed literature corpus; Rust/CUDA/PTX/header/build/test inputs remain. | Technical checks pass; repository history is not part of the prepared source release. |
| Working route | README gives the supported platform, native prerequisites, locked build, durable WAL, loopback trust profile, psql SQL/restart workflow, security profile, and limits. A 10-assertion GPU smoke automates it. | Working on the audit GPU. |
| Exact release proof | Hosted and trusted-GPU workflows, pinned Rust, source builder, focused recovery/pgwire/security gates, and non-vacuous GPU checks are present. | Exact committed archive, audit, and local tag remain after the declaration. |

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

The repository has 3,714 commits reachable from all refs. Git author records use three apparent variants of one
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

The literature-review corpus under `docs/archive/research/` is excluded because its papers, extracted text,
generated JSON, and image provenance is independent of program source and has not received publication clearance.
It is not a build, test, modification, or runtime input. Other historical project evidence under `docs/archive/`
remains because live architecture/status documents link to it; reviewed workstation paths were removed from the
distributed tree.

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
| Twice-built current pre-freeze source archive | Byte-identical outputs, SHA-256 `2e6bb1c462f4ae0c66edfbe0466fddc87c71b6af716e36d875ba7c4b5c45dd76`, 5,930,806 bytes; 27 PTX / 3 CUDA / 1 header; zero research or target entries. This hash binds temporary technical commit `a47f1d2a`, not the final rights-approved candidate. |
| Extracted current archive in a new Ubuntu 26.04 container | Clean locked release build passed in 3m39s with Rust 1.97.1, Clang/libclang 21.1.8, PostgreSQL client 18.6, the documented native packages, and no bindgen workaround. Product ownership and all 10 durable GPU SQL/SIGKILL/restart assertions passed against the container-built binary with the host RTX PRO 6000/driver 595.84 passed through. |
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
need Python 3 and the ownership guard needs ripgrep; both are now explicit prerequisites. The final exact committed
archive must repeat or validly carry this clean-container evidence after the rights-only documentation change.

The hosted workflow installs native prerequisites, pins Rust and policy-tool versions, checks the complete locked
workspace, and runs host-neutral tests. GPU absence is reported there. The manual trusted workflow instead fails
on missing NVIDIA hardware, bounds every gate, runs ownership, recovery, the complete pgwire matrix, all ignored
GPU tests, and the durable source-release smoke, and verifies a positive executed-test count.

No benchmark acceptance card has yet been attached to OSS-001. The candidate does not directly change a read
kernel, residency layout, result path, planner, benchmark harness, or release profile. It does upgrade the
production `imbl` persistent collections used by MVCC storage and engine value indexes. Layer 2 constructs its
resident fixture through those paths and links the changed implementation, so the canonical full report card is
**applicable** after the rights-approved candidate is frozen and passes its pre-card independent audit. Preserved
project APIs and call sites do not make this runtime dependency change inert. A non-canonical WIP quick screen
completed Sections A/B: its three batch-65,536 production-compact samples were 260,743,660 / 259,577,533 /
263,429,929 lookups/s (median 260,743,660; 2/3 at or above the 260M floor), and the script reported
`point_read_throughput_gate_status=pass`. Six unrelated resident GPU contexts were present and recorded by the
screen. The first attempt failed during the clean build because this unprovisioned host lacks Clang resource
headers; the completed rerun used the already disclosed temporary GCC-15 include workaround. The quick run is
diagnostic only and cannot close acceptance. The frozen candidate must run the full A/B/C card once under an
exclusive GPU and compare it to the preceding accepted comparable card. No performance claim is added by this
release. The retained accepted-card transcript named in `STATUS.md` is not available in this checkout, and the
accepted revision did not pin a Rust toolchain. Its exact compiler identity must be recovered before claiming
comparability with the newly pinned Rust 1.97.1 candidate; if it cannot be recovered or differs materially, rerun
the accepted base revision in the same environment before the candidate card. The audit GPU currently has unrelated
resident processes, so neither canonical run may start until an exclusive window is available.

## Remaining freeze sequence

1. Repeat affected static/secret checks after the recorded declaration; the WIP quick performance screen is
   complete.
2. Stage one candidate with no unstaged or untracked drift and create a local DCO-signed release commit.
3. Run the mandatory independent read-only pre-card acceptance audit and repair/re-audit any finding.
4. Run the canonical full report card once on that exact candidate under an exclusive GPU and have the auditor
   verify its completeness, provenance, and accepted-baseline comparison.
5. Build the source archive twice from that exact commit, compare bytes, extract it into a clean provisioned
   environment, build with `--locked`, and run the documented durable GPU/pgwire route.
6. Record the commit, archive SHA-256/size, test counts, and audit result; create the local
   `v0.1.0-alpha.1` tag. Do not push, publish, or change visibility.
