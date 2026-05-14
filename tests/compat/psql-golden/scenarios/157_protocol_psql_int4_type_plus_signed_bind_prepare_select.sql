\echo === psql int4 ddl and plus-signed literals ===
CREATE TABLE plus_int4_people (id INT4, name pg_catalog.text);
INSERT INTO plus_int4_people (id, name) VALUES (-1, 'Minus'), (0, 'Zero'), (1, 'Plus');
SELECT name FROM plus_int4_people WHERE id = +1;
SELECT name FROM plus_int4_people WHERE +0 = id;
SELECT name FROM plus_int4_people WHERE id = $1 \bind +1 \g
PREPARE plus_int4_lookup(int4) AS SELECT name FROM plus_int4_people WHERE id = $1;
EXECUTE plus_int4_lookup(+1);
EXECUTE plus_int4_lookup(+0);
DEALLOCATE PREPARE plus_int4_lookup;
SELECT name FROM plus_int4_people WHERE id = -1;
