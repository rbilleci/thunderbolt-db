\echo === psql extended bind crosstabview ===
CREATE TABLE ext_crosstab_people (id INT, name TEXT, bucket TEXT);
INSERT INTO ext_crosstab_people (id, name, bucket) VALUES (1, 'Ada', 'east'), (2, 'Linus', 'west'), (3, 'Grace', 'east');
SELECT bucket, name, id FROM ext_crosstab_people WHERE id >= $1 ORDER BY id \bind 1 \crosstabview
SELECT name FROM ext_crosstab_people WHERE id = 2;
