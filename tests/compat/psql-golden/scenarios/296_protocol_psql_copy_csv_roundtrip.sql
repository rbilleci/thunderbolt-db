\echo === psql copy csv import export recovery ===
CREATE TABLE copy_csv_people (id INT, name TEXT);
COPY copy_csv_people FROM STDIN WITH CSV;
1,Ada
2,"Grace, ""Hopper"""
3,"Line\tLiteral"
\.
SELECT id, name FROM copy_csv_people ORDER BY id;
COPY copy_csv_people TO STDOUT WITH CSV;
SELECT name FROM copy_csv_people WHERE id = 2;
