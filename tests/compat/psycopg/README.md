# psycopg compatibility smoke

This suite exercises the supported PostgreSQL wire/SQL subset through the Python `psycopg` client.

Run it from this directory:

```sh
./run.sh
```

The smoke installs `psycopg` and `psycopg-pool` into a temporary dependency directory, builds and starts the canonical `thunderbolt-db-server`, then covers startup, simple query, prepared/bound `SELECT`, empty results, explicit unsupported broader `COPY ... WITH CSV` options, same-session recovery, one-connection pool reuse, and reconnect behavior.
