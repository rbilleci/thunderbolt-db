# Third-party notices

This repository contains project-authored source and build descriptions for software obtained separately by
Cargo or supplied by the operating environment. Those components retain their own copyrights and license terms.
The project does not relicense them under GPL-3.0-only.

The exact Rust dependency versions for this candidate are fixed by `Cargo.lock`. The generated
[`dependency inventory`](docs/design/open-source-dependencies.tsv) records every external package and its declared
license expression. The reviewed `deny.toml` policy checks the locked graph's advisories, licenses, and sources;
release evidence records the inventory comparison against Cargo metadata.

Components needing specific attention include:

| Component | Use and source | License/notices |
|---|---|---|
| `pg_query 6.1.1` / `libpg_query` | PostgreSQL 17.4 parser dependency built from source by the `pg_query` crate | MIT wrapper (Paul Mason and Duboce Labs/pganalyze); BSD-3-Clause libpg_query code; PostgreSQL License for PostgreSQL-derived sources; BSD-2-Clause protobuf-c and xxHash. See `THIRD_PARTY_LICENSES/`. |
| `imbl 7.0.2`, `imbl-sized-chunks 0.2.0` | Persistent collections fetched by Cargo | MPL-2.0 or later. The unmodified upstream files and notices remain subject to the MPL; no Exhibit B incompatibility notice was found in the inspected crate sources. |
| `aws-lc-rs 1.17.0`, `aws-lc-sys 0.41.0` | TLS cryptography implementation fetched and built by Cargo | ISC/Apache-2.0 and the additional permissive terms reproduced in `THIRD_PARTY_LICENSES/AWS-LC.txt`. |
| `tikv-jemallocator 0.6.1`, `tikv-jemalloc-sys 0.6.1` / jemalloc | Server allocator fetched and built by Cargo | MIT/Apache-2.0 wrapper terms and the BSD-style jemalloc terms reproduced in `THIRD_PARTY_LICENSES/jemalloc.txt`. |
| NVIDIA display driver and CUDA driver API | Required external runtime providing `libcuda.so.1`; dynamically loaded by the engine | Proprietary NVIDIA software supplied and licensed separately by the operator. It is not included in the source release and is not covered by the project license. |
| NVIDIA CUDA compiler (`nvcc`) | Optional external tool used to regenerate checked-in PTX from the preferred-form `.cu` and `.cuh` sources | Proprietary NVIDIA SDK tool supplied and licensed separately by the developer. It is not needed to build or run the checked-in PTX. |

Permissive Rust dependencies fetched by Cargo retain the license files and copyright notices in their upstream
source packages. Anyone distributing binaries must review the exact target-specific dependency graph and include
all notices required by those source packages and the platform components they redistribute. This file describes
the source-release boundary; it is not a substitute for a binary-distribution compliance review.

No NVIDIA driver, CUDA Toolkit binary, CUDA header, or NVIDIA sample source is included in this repository.
The project CUDA kernels and PTX are project source/output; their preferred editable `.cu`/`.cuh` sources and
reproduction commands are included alongside the PTX.

The historical literature-review corpus under `docs/archive/research/` is excluded from the versioned source
archive. It is not program source or a build input. The remaining release documentation is project-authored prose
or retains the notices stated in its files.

NVIDIA and CUDA are trademarks or registered trademarks of NVIDIA Corporation. This project is not affiliated
with or endorsed by NVIDIA. PostgreSQL is a trademark of the PostgreSQL Community Association of Canada;
compatibility references are descriptive only.
