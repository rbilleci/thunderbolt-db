\echo === session reset probes ===
RESET ALL;
DISCARD ALL;
DEALLOCATE ALL;
UNLISTEN *;
\echo === sql prepare execute flow ===
PREPARE golden_stmt(int) AS SELECT $1 + 10 AS plus_ten;
EXECUTE golden_stmt(5);
DEALLOCATE golden_stmt;
