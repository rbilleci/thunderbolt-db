\echo === session reset probes ===
SELECT pg_advisory_unlock_all();
CLOSE ALL;
RESET ALL;
DISCARD ALL;
DEALLOCATE ALL;
UNLISTEN *;
\echo === sql prepare execute flow ===
PREPARE golden_stmt(int) AS SELECT $1 + 10 AS plus_ten;
EXECUTE golden_stmt(5);
DEALLOCATE golden_stmt;
