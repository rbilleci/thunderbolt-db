\echo === psql extended zero-padded nonzero placeholders ===
CREATE TABLE ext_zero_padded_nonzero_people (id INT, name TEXT);
INSERT INTO ext_zero_padded_nonzero_people (id, name) VALUES (1, 'Ada'), (2, 'Linus'), (3, 'Grace');
SELECT id, name FROM ext_zero_padded_nonzero_people WHERE id = $01 OR id = $002 ORDER BY id LIMIT $0003 \bind 1 2 2 \g
