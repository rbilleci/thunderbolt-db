# Replication follower rename migration

The follower Deployments and Services now use the `thunderbolt-db` resource
names, labels, selectors, and `thunderbolt-db-replication-service:local`
image. The `GPU_DB_REPLICATION_*` environment-variable names are unchanged.

Kubernetes Deployment selectors are immutable. A live
`gpu-db-replication-follower-*` Deployment cannot be patched to the new
`thunderbolt-db-follower-id` selector. For each follower, stop the old
Deployment before creating the renamed replacement so two processes never
serve the same follower ID. Preserve any operator-managed PVCs, Secrets, and
image pull configuration; this local manifest does not declare them.

```sh
kubectl scale deployment/gpu-db-replication-follower-2 --replicas=0
kubectl scale deployment/gpu-db-replication-follower-3 --replicas=0
kubectl delete deployment/gpu-db-replication-follower-2 gpu-db-replication-follower-3
kubectl apply -f k8s/replication-service/follower-services.yml
kubectl rollout status deployment/thunderbolt-db-replication-follower-2
kubectl rollout status deployment/thunderbolt-db-replication-follower-3
# Update append-entry clients to the new Service DNS names before this removal.
kubectl delete service gpu-db-replication-follower-2 gpu-db-replication-follower-3
```

Update append-entry clients to use the new Service DNS names
`thunderbolt-db-replication-follower-2` and
`thunderbolt-db-replication-follower-3` before removing the old Services.
