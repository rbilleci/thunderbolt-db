# asyncpg compatibility smoke

This suite exercises the supported PostgreSQL wire/SQL subset through the Python `asyncpg` client.

Run it from this directory:

```sh
./run.sh
```

The smoke installs `asyncpg` into a temporary dependency directory, builds and starts `gpu-db-server`, then covers startup, simple query, prepared/bound `SELECT`, empty results, explicit unsupported `COPY ... FROM STDIN WITH CSV`, same-session recovery, pooled query, and reconnect behavior.
