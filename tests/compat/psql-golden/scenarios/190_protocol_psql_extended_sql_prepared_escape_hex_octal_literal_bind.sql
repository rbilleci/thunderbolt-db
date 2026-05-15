\echo === psql extended sql prepared escape hex octal literal bind ===
CREATE TABLE ext_sql_exec_escape_hex_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_escape_hex_people (id, name) VALUES (1, 'Ada Lovelace'), (2, 'Grace Hopper'), (3, 'Linus');
PREPARE ext_sql_exec_escape_hex_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_escape_hex_people WHERE name = $1 AND id = $2;
EXECUTE ext_sql_exec_escape_hex_lookup(E'Ada\x20Lovelace', $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_escape_hex_lookup(E'Ada\x20Lovelace', $1) \bind 1 \g
EXECUTE ext_sql_exec_escape_hex_lookup(E'Grace\040Hopper', $1) \bind 2 \g
EXECUTE ext_sql_exec_escape_hex_lookup(E'Ada\x20Lovelace', $1) \bind not-an-int \g
EXECUTE ext_sql_exec_escape_hex_lookup(E'bad\xzz', $1) \bind 1 \g
EXECUTE ext_sql_exec_escape_hex_lookup($1, 3) \bind Linus \g
DEALLOCATE PREPARE ext_sql_exec_escape_hex_lookup;
SELECT name FROM ext_sql_exec_escape_hex_people WHERE id = 1;
