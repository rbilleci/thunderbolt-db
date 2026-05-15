\echo === psql cursor sql execute negative limit gdesc ===
CREATE TABLE cursor_exec_negative_gdesc_people (id INT, name TEXT);
INSERT INTO cursor_exec_negative_gdesc_people (id, name) VALUES (1, 'Ada'), (2, 'Ada'), (3, 'Grace');
PREPARE cursor_exec_negative_gdesc_lookup(int4) AS SELECT id, name FROM cursor_exec_negative_gdesc_people ORDER BY id LIMIT $1;
DECLARE cursor_exec_negative_desc CURSOR FOR EXECUTE cursor_exec_negative_gdesc_lookup($1) \bind -1 \gdesc
DECLARE cursor_exec_negative_bad CURSOR FOR EXECUTE cursor_exec_negative_gdesc_lookup($1) \bind -1 \g
DECLARE cursor_exec_negative_ok CURSOR FOR EXECUTE cursor_exec_negative_gdesc_lookup($1) \bind 1 \g
FETCH ALL FROM cursor_exec_negative_ok;
CLOSE cursor_exec_negative_ok;
DEALLOCATE PREPARE cursor_exec_negative_gdesc_lookup;
SELECT name FROM cursor_exec_negative_gdesc_people WHERE id = 3;
