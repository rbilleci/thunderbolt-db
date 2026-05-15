\echo === psql cursor sql execute describe literal error ===
CREATE TABLE cursor_exec_describe_error_people (id INT, name TEXT);
INSERT INTO cursor_exec_describe_error_people (id, name) VALUES (1, 'Ada'), (2, 'Linus');
PREPARE cursor_exec_describe_error_lookup(int4) AS SELECT name FROM cursor_exec_describe_error_people WHERE id = $1;
DECLARE cursor_exec_describe_error CURSOR FOR EXECUTE cursor_exec_describe_error_lookup('not-an-int') \gdesc
DECLARE cursor_exec_describe_ok CURSOR FOR EXECUTE cursor_exec_describe_error_lookup(2);
FETCH 1 FROM cursor_exec_describe_ok;
CLOSE cursor_exec_describe_ok;
DEALLOCATE PREPARE cursor_exec_describe_error_lookup;
SELECT name FROM cursor_exec_describe_error_people WHERE id = 1;
