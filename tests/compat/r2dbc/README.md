# R2DBC compatibility smoke

This checked smoke runs `thunderbolt-db-server` on an ephemeral local port and verifies
the supported PostgreSQL wire/SQL subset through the mainstream PostgreSQL
R2DBC driver with TLS disabled.

It covers startup/connect, a simple query, an `int4`/`text` table workflow,
prepared/bound query results, empty results, explicit unsupported COPY option
handling with same-connection recovery, connection-factory reuse, and reconnect.
The driver keeps its default extension autodetection enabled, so startup also proves
its exact `pg_catalog.pg_type` query through the canonical GPU catalog path.
R2DBC pooling is not claimed by this smoke.
