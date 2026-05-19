\echo === catalog psql list bootstrap language ===
\pset pager off
\dL
SELECT tableoid, oid, lanname, lanpltrusted, lanplcallfoid, laninline, lanvalidator, lanacl, acldefault('l', lanowner) AS acldefault, lanowner
  FROM pg_language
 WHERE lanispl
 ORDER BY oid;
