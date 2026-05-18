# node-postgres compatibility smoke

This suite exercises the supported PostgreSQL wire/SQL subset through the JavaScript `pg` client.

Run it from this directory:

```sh
npm ci
npm test
```

The smoke builds and starts `gpu-db-server`, then covers startup, simple query, named prepared/bound `SELECT`, empty results, explicit unsupported broader `COPY ... WITH CSV` options, same-session recovery, pooled query, and reconnect behavior.
