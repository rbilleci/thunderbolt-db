# node-postgres compatibility smoke

This suite exercises the supported PostgreSQL wire/SQL subset through the JavaScript `pg` client.

Run it from this directory:

```sh
npm ci
npm test
```

The smoke builds and starts the canonical facade-backed `gpu-db-engine-server`, then covers startup
and parsed BackendKeyData, simple query, named prepared/bound `SELECT`, empty results, explicit
unsupported broader `COPY ... WITH CSV` options, same-session recovery, pooled query, and reconnect
behavior.
