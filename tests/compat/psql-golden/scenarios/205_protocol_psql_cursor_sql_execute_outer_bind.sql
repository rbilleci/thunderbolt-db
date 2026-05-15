\echo === psql cursor sql execute outer bind ===
CREATE TABLE cursor_exec_bind_people (id INT, name TEXT);
INSERT INTO cursor_exec_bind_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
PREPARE cursor_exec_bind_lookup(int4) AS SELECT id, name FROM cursor_exec_bind_people WHERE id >= $1 ORDER BY id;
DECLARE cursor_exec_bind_desc CURSOR FOR EXECUTE cursor_exec_bind_lookup($1) \bind 2 \gdesc
DECLARE cursor_exec_bind_ok CURSOR FOR EXECUTE cursor_exec_bind_lookup($1) \bind 2 \g
FETCH 1 FROM cursor_exec_bind_ok;
CLOSE cursor_exec_bind_ok;
DEALLOCATE PREPARE cursor_exec_bind_lookup;
SELECT name FROM cursor_exec_bind_people WHERE id = 1;
