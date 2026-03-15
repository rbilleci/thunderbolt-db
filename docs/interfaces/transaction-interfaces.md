# Transaction Interfaces (Draft)

## TxnManager
- `begin(isolation)`
- `snapshot(txn_id)`
- `commit(txn_id)`
- `abort(txn_id)`

## Visibility
- `visible(version, snapshot) -> bool`
- must be consistent across CPU and GPU paths
