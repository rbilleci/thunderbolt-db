# Replication follower rename migration

The template unit is now `thunderbolt-db-replication-follower@.service`. The
old `gpu-db-replication-follower@.service` unit, `/etc/gpu-db` configuration
directory, `/var/lib/gpu-db` state directory, and `/usr/share/doc/gpu-db`
documentation path are retired names.

Migrate a stopped follower instance before installing the renamed unit. Replace
`2` with the follower instance being moved, and repeat for every instance:

```sh
sudo systemctl disable --now gpu-db-replication-follower@2.service
sudo install -d -m 0750 /etc/thunderbolt-db /var/lib/thunderbolt-db
sudo cp -a /etc/gpu-db/replication-follower@2.env /etc/thunderbolt-db/
sudo cp -a /var/lib/gpu-db/. /var/lib/thunderbolt-db/
sudo install -m 0644 systemd/replication-follower/thunderbolt-db-replication-follower@.service \
  /etc/systemd/system/thunderbolt-db-replication-follower@.service
sudo systemctl daemon-reload
sudo systemctl enable --now thunderbolt-db-replication-follower@2.service
sudo systemctl status thunderbolt-db-replication-follower@2.service
```

Keep the old unit and paths until the renamed follower has recovered and
replicated its durable prefix. Do not run the old and new units for the same follower ID concurrently. The `GPU_DB_REPLICATION_*` variable names in the environment files are unchanged compatibility configuration.
