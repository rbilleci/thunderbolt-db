//! Shared SQL parse diagnostics.

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("empty command")]
    Empty,
    #[error("unsupported command: {0}")]
    Unsupported(String),
    #[error("invalid SET syntax; expected: SET key=value or SET key TO value")]
    InvalidSet,
    #[error("invalid DEL/DELETE syntax; expected: DEL key or DELETE [FROM] key")]
    InvalidDel,
    #[error("invalid GET syntax; expected: GET key")]
    InvalidGet,
    #[error("invalid relational SQL syntax; supported subset: CREATE TABLE name (...), CREATE [UNIQUE] INDEX name ON table (column), DROP INDEX [IF EXISTS] name, INSERT INTO name (...) VALUES (...), UPDATE name SET column = literal [, ...] WHERE column (=|<|<=|>|>=) literal | column BETWEEN literal AND literal | column IN (literal, ...) | text_column LIKE 'prefix%' [AND ...] [OR ...], DELETE FROM name WHERE column (=|<|<=|>|>=) literal | column BETWEEN literal AND literal | column IN (literal, ...) | text_column LIKE 'prefix%' [AND ...] [OR ...], SELECT [DISTINCT] columns|COUNT(*)|SUM(int4_column)|AVG(int4_column)|MIN(column)|MAX(column)|column, COUNT(*)|column, SUM(int4_column)|column, AVG(int4_column)|column, MIN(column)|column, MAX(column) FROM name [WHERE column (=|<|<=|>|>=) literal | column BETWEEN literal AND literal | column IN (literal, ...) | text_column LIKE 'prefix%' [AND ...] [OR ...]] [GROUP BY column] [HAVING grouped_column|count|sum|avg|min|max (=|<|<=|>|>=) literal [AND ...] [OR ...]] [ORDER BY selected_column|count|sum|avg|min|max [ASC|DESC]] [LIMIT n] [OFFSET n]")]
    InvalidRelationalSql,
    #[error("invalid input syntax for type {ty}: \"{input}\"")]
    InvalidTextRepresentation { ty: &'static str, input: String },
    #[error("invalid date/time format: \"{input}\"")]
    InvalidDatetimeFormat { input: String },
    #[error("date/time field value out of range: \"{input}\"")]
    DatetimeFieldOverflow { input: String },
    #[error("value \"{input}\" is out of range for type {ty}")]
    NumericValueOutOfRange { ty: &'static str, input: String },
    #[error("invalid SQL parameter reference")]
    InvalidParameterReference,
    #[error("SQL parameter count mismatch: expected {expected}, got {actual}")]
    InvalidParameterCount { expected: usize, actual: usize },
    #[error("LIMIT must not be negative")]
    NegativeLimit,
    #[error("OFFSET must not be negative")]
    NegativeOffset,
    #[error("invalid RESET/DISCARD/DEALLOCATE/CLOSE/LISTEN/NOTIFY/UNLISTEN syntax; expected: RESET ALL|ROLE|AUTHORIZATION|AUTH|SESSION AUTHORIZATION[ [TO] DEFAULT]|SESSION AUTH[ [TO] DEFAULT], DISCARD {{ALL|TEMP|TEMPORARY|TEMP TABLES|TEMPORARY TABLES|PLANS|SEQUENCES}}, DEALLOCATE {{ALL|name|PREPARE|PREPARED name}}, CLOSE {{ALL|name}}, LISTEN channel, NOTIFY channel[, payload], or UNLISTEN [*|ALL|channel]")]
    InvalidReset,
}
