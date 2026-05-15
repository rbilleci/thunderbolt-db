\echo === psql extended sql prepared comment adjacent literal bind ===
CREATE TABLE ext_sql_exec_comment_adjacent_people (id INT, name TEXT);
INSERT INTO ext_sql_exec_comment_adjacent_people (id, name) VALUES (1, 'Ada Lovelace'), (2, 'Grace Hopper'), (3, 'Linus');
PREPARE ext_sql_exec_comment_adjacent_lookup(text, int4) AS SELECT id, name FROM ext_sql_exec_comment_adjacent_people WHERE name = $1 AND id >= $2 ORDER BY id;
EXECUTE ext_sql_exec_comment_adjacent_lookup('Ada' /* keep newline
inside comment */ ' Lovelace', $1) \bind 1 \gdesc
EXECUTE ext_sql_exec_comment_adjacent_lookup('Ada' /* keep newline
inside comment */ ' Lovelace', $1) \bind 1 \g
EXECUTE ext_sql_exec_comment_adjacent_lookup('Grace' -- line comment preserves newline
' Hopper', $1) \bind 2 \g
EXECUTE ext_sql_exec_comment_adjacent_lookup('Ada' /* no newline */ 'Lovelace', $1) \bind 1 \g
DEALLOCATE PREPARE ext_sql_exec_comment_adjacent_lookup;
SELECT name FROM ext_sql_exec_comment_adjacent_people WHERE id = 3;
