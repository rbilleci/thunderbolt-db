# JDBC compatibility smoke

This checked smoke runs the canonical `thunderbolt-db-server` on an ephemeral local port and verifies
the supported PostgreSQL wire/SQL subset through the PostgreSQL JDBC driver
with TLS disabled.

It covers startup/connect, a simple query, an `int4`/`text` table workflow,
prepared/bound query results, empty results, explicit unsupported COPY option
handling with same-session recovery, and reconnect. R2DBC remains out of scope.
