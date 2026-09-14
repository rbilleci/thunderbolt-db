# Contributing

Thunderbolt DB is an experimental GPU-native OLTP database. Read [`AGENTS.md`](AGENTS.md),
[`docs/CHARTER.md`](docs/CHARTER.md), and the active work in [`docs/PLAN.md`](docs/PLAN.md) before changing the
engine. The GPU data-plane boundary, WAL-before-visibility rule, and acceptance gates apply to human and automated
contributions alike.

Open an issue before a large change so its production route, supported SQL shape, and verification plan are clear.
Keep one pull request focused on one PLAN milestone. A feature must connect to the production call graph and remove
the superseded live route in the same candidate. CPU relational execution is a test oracle or explicitly tracked
bootstrap debt, not a product implementation.

For a normal change:

1. Branch from the current default branch and make the smallest coherent change that closes a production-visible
   assertion.
2. Add focused correctness coverage. Device behavior requires non-vacuous GPU and NULL/3VL coverage, bounded test
   timeouts, and the HAZARD protocol described in `AGENTS.md`.
3. Run the affected crate checks and tests, `git diff --check`, and the wider gates required by `AGENTS.md`.
   Read/residency/result-path changes also use the benchmark report-card workflow.
4. Explain the behavior, evidence, and remaining limits in the pull request. Do not report a PLAN milestone as
   accepted until its complete integrated candidate passes the independent audit.

All project contributions are offered under GPL-3.0-only with the same CUDA Driver Additional Permission in
[`CUDA_EXCEPTION`](CUDA_EXCEPTION). Contributions must include a Developer Certificate of Origin sign-off in every
commit:

```text
Signed-off-by: Your Name <your-email@example.com>
```

Use `git commit -s` to add it. The sign-off certifies the terms in [`DCO`](DCO). Preserve copyright and license
notices on third-party material, identify its source and license in the pull request, and do not submit code or data
you do not have the right to publish.

Report vulnerabilities privately as described in [`SECURITY.md`](SECURITY.md).
