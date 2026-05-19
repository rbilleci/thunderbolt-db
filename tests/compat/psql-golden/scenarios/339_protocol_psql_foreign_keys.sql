\echo === bounded foreign keys ===
CREATE TABLE fk_customers (id INT PRIMARY KEY, name TEXT);
CREATE TABLE fk_orders (id INT PRIMARY KEY, customer_id INT);
INSERT INTO fk_customers (id, name) VALUES (1, 'Ada'), (2, 'Grace');
INSERT INTO fk_orders (id, customer_id) VALUES (10, 1);
ALTER TABLE ONLY public.fk_orders ADD CONSTRAINT fk_orders_customer_fk FOREIGN KEY (customer_id) REFERENCES public.fk_customers(id);
INSERT INTO fk_orders (id, customer_id) VALUES (11, 99);
COPY fk_orders (id, customer_id) FROM STDIN;
12	99
\.
UPDATE fk_orders SET customer_id = 99 WHERE id = 10;
DELETE FROM fk_customers WHERE id = 1;
SELECT table_schema, table_name, constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_schema = 'public' ORDER BY table_name, constraint_name;
SELECT table_schema, table_name, column_name, constraint_name, ordinal_position FROM information_schema.key_column_usage WHERE table_schema = 'public' ORDER BY table_name, ordinal_position;
SELECT n.nspname, c.relname, con.conname, con.contype
FROM pg_catalog.pg_constraint con
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
COMMENT ON CONSTRAINT fk_orders_customer_fk ON fk_orders IS 'bounded foreign key';
SELECT n.nspname, c.relname, con.conname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_constraint con ON con.oid = d.objoid
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
\d+ fk_orders
\d+ fk_customers
ALTER TABLE ONLY fk_orders RENAME CONSTRAINT fk_orders_customer_fk TO fk_orders_customer_ref_fk;
COMMENT ON CONSTRAINT fk_orders_customer_ref_fk ON fk_orders IS 'renamed foreign key';
SELECT table_schema, table_name, constraint_name, constraint_type FROM information_schema.table_constraints WHERE table_schema = 'public' ORDER BY table_name, constraint_name;
SELECT n.nspname, c.relname, con.conname, d.description
FROM pg_catalog.pg_description d
JOIN pg_catalog.pg_constraint con ON con.oid = d.objoid
JOIN pg_catalog.pg_class c ON c.oid = con.conrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
ORDER BY c.relname, con.conname;
ALTER TABLE ONLY fk_orders DROP CONSTRAINT fk_orders_customer_ref_fk;
DELETE FROM fk_customers WHERE id = 1;
SELECT id, customer_id FROM fk_orders ORDER BY id;
ALTER TABLE ONLY fk_orders ADD CONSTRAINT fk_orders_missing_fk FOREIGN KEY (customer_id) REFERENCES fk_customers(name);
ALTER TABLE ONLY fk_orders ADD CONSTRAINT fk_orders_no_unique_fk FOREIGN KEY (customer_id) REFERENCES fk_customers(id) ON DELETE CASCADE;
ALTER TABLE ONLY fk_orders ADD CONSTRAINT fk_orders_self_fk FOREIGN KEY (customer_id) REFERENCES fk_orders(id);
INSERT INTO fk_customers (id, name) VALUES (1, 'Ada restored');
ALTER TABLE ONLY fk_orders ADD CONSTRAINT fk_orders_customer_fk FOREIGN KEY (customer_id) REFERENCES fk_customers(id);
DROP TABLE fk_customers;
ALTER TABLE ONLY fk_orders DROP CONSTRAINT fk_orders_customer_fk;
DROP TABLE fk_orders, fk_customers;
