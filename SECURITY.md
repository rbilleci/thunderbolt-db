# Security policy

This repository is an experimental database engine and has no supported production release yet. The current
source release is intended for evaluation on isolated systems with non-sensitive data. Security fixes target the
current default branch; no backport window is promised before a supported release is declared.

Report a suspected vulnerability through GitHub's private vulnerability reporting form:

<https://github.com/mvsm-prometheus/gpu-database-engine/security/advisories/new>

Include the affected revision, configuration, impact, reproduction steps, and any suggested mitigation. Do not
open a public issue with exploit details or secrets. If the private form is unavailable, open a public issue that
only asks the maintainers to enable or provide a private security channel; omit all sensitive details.

The default `local-dev` server profile listens on loopback and uses trust authentication without TLS. It is only
for local evaluation. Any non-loopback deployment must use the explicit production TLS and SCRAM-SHA-256 profile,
network isolation, and non-sensitive evaluation data. This policy does not make the experimental release suitable
for production use.
