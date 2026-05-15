\echo === psql extended bind crosstabview invalid recovery ===
CREATE TABLE ext_bind_crosstab_invalid_people (id INT, name TEXT, bucket TEXT);
INSERT INTO ext_bind_crosstab_invalid_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
SELECT bucket, name, id FROM ext_bind_crosstab_invalid_people WHERE id >= $1 ORDER BY id \bind not_an_int \crosstabview
SELECT bucket, name, id FROM ext_bind_crosstab_invalid_people WHERE id >= $1 ORDER BY id \bind 2 \crosstabview
