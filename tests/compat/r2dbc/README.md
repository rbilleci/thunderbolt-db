# R2DBC compatibility smoke

This checked smoke runs `gpu-db-server` on an ephemeral local port and verifies
the supported PostgreSQL wire/SQL subset through the mainstream PostgreSQL
R2DBC driver with TLS disabled.

It covers startup/connect, a simple query, an `int4`/`text` table workflow,
prepared/bound query results, empty results, explicit unsupported COPY option
handling with same-connection recovery, connection-factory reuse, and reconnect.
R2DBC pooling is not claimed by this smoke.
