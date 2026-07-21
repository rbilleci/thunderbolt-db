# asyncpg compatibility smoke

This suite exercises the supported PostgreSQL wire/SQL subset through the Python `asyncpg` client.

Run it from this directory:

```sh
./run.sh
```

The smoke installs `asyncpg` into a temporary dependency directory, builds and starts the canonical `gpu-db-engine-server`, then covers startup, simple query, prepared/bound `SELECT`, empty results, explicit unsupported broader `COPY ... WITH CSV` options, same-session recovery, pooled query with asyncpg's default reset cleanup SQL, pool reuse, and reconnect behavior.
