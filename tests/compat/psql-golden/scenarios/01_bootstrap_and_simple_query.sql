\echo === connect and simple query ===
SELECT 1 AS one;
\echo === session probes ===
SHOW client_encoding;
SELECT current_schema();
\echo === tx begin/commit/rollback ===
BEGIN;
SELECT 2 AS in_tx;
COMMIT;
BEGIN;
SELECT 3 AS rolled_back;
ROLLBACK;
\echo === deterministic error path ===
SELECT * FROM definitely_missing_relation_for_golden;
