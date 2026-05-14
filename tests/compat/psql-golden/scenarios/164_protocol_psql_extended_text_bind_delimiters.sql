\echo === psql extended text bind delimiter literals ===
CREATE TABLE ext_text_delimiter_people (id INT, name TEXT);
INSERT INTO ext_text_delimiter_people (id, name) VALUES
  (1, 'Ada Lovelace'),
  (2, 'semi;colon'),
  (3, 'line -- comment'),
  (4, 'block /* comment */ marker'),
  (5, 'comma,value');
SELECT id FROM ext_text_delimiter_people WHERE name = $1 \bind 'Ada Lovelace' \g
SELECT id FROM ext_text_delimiter_people WHERE name = $1 \bind 'semi;colon' \g
SELECT id FROM ext_text_delimiter_people WHERE name = $1 \bind 'line -- comment' \g
SELECT id FROM ext_text_delimiter_people WHERE name = $1 \bind 'block /* comment */ marker' \g
SELECT id FROM ext_text_delimiter_people WHERE name = $1 \bind 'comma,value' \g
SELECT name FROM ext_text_delimiter_people WHERE id = 5;
