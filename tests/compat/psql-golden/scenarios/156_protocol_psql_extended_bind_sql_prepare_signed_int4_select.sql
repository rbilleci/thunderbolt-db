\echo === psql signed int4 parameters ===
CREATE TABLE signed_int4_people (id INT, name TEXT);
INSERT INTO signed_int4_people (id, name) VALUES (-1, 'Minus'), (0, 'Zero'), (2, 'Plus');
SELECT name FROM signed_int4_people WHERE id = -1;
SELECT name FROM signed_int4_people WHERE id = $1 \bind -1 \g
PREPARE signed_int4_lookup(int4) AS SELECT name FROM signed_int4_people WHERE id = $1;
EXECUTE signed_int4_lookup(-1);
EXECUTE signed_int4_lookup(0);
DEALLOCATE PREPARE signed_int4_lookup;
SELECT name FROM signed_int4_people WHERE id = 2;
