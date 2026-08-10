//! INSERT qualification instrumentation at the facade's one canonical submission boundary.
//!
//! This leaf owns only feature-gated probe accessors and timing around the existing text/prepared
//! submission flow. It delegates parsed commands straight back to the parent `submit_parsed`, so
//! instrumentation never creates a second execution or transaction-admission path.

#[cfg(feature = "probe-timing")]
use std::time::Instant;

#[cfg(feature = "probe-timing")]
pub use gpu_db_engine::{InsertProbeConfig, InsertProbeSnapshot};
#[cfg(feature = "probe-timing")]
use gpu_db_sql::Command;
use gpu_db_sql::{ParseError, ParsedCommand};

#[cfg(feature = "probe-timing")]
use super::CommandTag;
use super::{
    map_parse_error, prepared, submit_general_select_text_inner, submit_parsed,
    BoundPreparedStatement, DbError, QueryOutcome, SharedEngine, SharedSession,
};

impl SharedEngine {
    /// Feature-gated, protocol-neutral access to engine-local INSERT qualification counters.
    #[cfg(feature = "probe-timing")]
    pub fn insert_probe_snapshot(&self) -> InsertProbeSnapshot {
        self.engine.insert_probe_snapshot()
    }

    /// Feature-gated resolved configuration accompanying an INSERT qualification record.
    #[cfg(feature = "probe-timing")]
    pub fn insert_probe_config(&self) -> InsertProbeConfig {
        self.engine.insert_probe_config()
    }

    /// Feature-gated, read-only evidence that a single named GPU index covers its resident rows.
    /// This is intentionally a façade pass-through: protocol integration tests must not inspect
    /// engine internals or create a second index authority.
    #[cfg(feature = "probe-timing")]
    pub fn relational_named_index_covered_rows(&self, table_name: &str) -> Option<usize> {
        self.engine.relational_named_index_covered_rows(table_name)
    }

    /// Feature-gated qualification precondition for a real protocol fixture. This invokes the
    /// existing engine-owned named-index publication operation after the fixture has admitted its
    /// seed row through pgwire; it exposes neither a writable engine reference nor a new SQL,
    /// index, or publication authority.
    #[cfg(feature = "probe-timing")]
    pub fn publish_relational_resident_indexes_for_qualification(
        &self,
        table_name: &str,
    ) -> Result<(), String> {
        self.engine
            .publish_relational_resident_indexes(table_name)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    /// Feature-gated server seam for exact raw simple-query bytes. The server calls this only
    /// after a Query message has produced exactly one successful INSERT.
    #[cfg(feature = "probe-timing")]
    #[doc(hidden)]
    pub fn record_insert_probe_successful_raw_simple_query_bytes(&self, source_bytes: u64) {
        self.engine
            .record_insert_probe_successful_raw_simple_query_bytes(source_bytes);
    }
}

/// Execute a bound statement through the existing prepared owner and attribute only successful
/// INSERT service. This wrapper never parses, validates, or admits a second statement.
pub(super) fn submit_prepared_inner(
    shared: &SharedEngine,
    session: &mut SharedSession,
    bound: &BoundPreparedStatement,
) -> Result<QueryOutcome, DbError> {
    #[cfg(feature = "probe-timing")]
    let probe_bind_started = bound.is_insert().then(Instant::now);
    let result = prepared::submit_prepared_inner(shared, session, bound);
    #[cfg(feature = "probe-timing")]
    if let Some(started) = probe_bind_started {
        let elapsed = started.elapsed().as_nanos() as u64;
        shared
            .engine
            .record_insert_probe_facade_parse_bind_nanos(elapsed);
        if matches!(
            &result,
            Ok(QueryOutcome::Command {
                tag: CommandTag::Insert,
                ..
            })
        ) {
            shared
                .engine
                .record_insert_probe_successful_prepared_service_nanos(elapsed);
        }
    }
    result
}

/// Execute text through the concurrent shared engine while preserving one transaction owner per
/// session. The parser/fallback and final `submit_parsed` delegation are the pre-existing single
/// text-submission path; this leaf merely keeps its INSERT timing at that exact seam.
pub(super) fn submit_text_inner(
    shared: &SharedEngine,
    session: &mut SharedSession,
    sql: &str,
) -> Result<QueryOutcome, DbError> {
    #[cfg(feature = "probe-timing")]
    let probe_statement_started = Instant::now();
    let parsed_result = ParsedCommand::parse_allowing_catalog(sql);
    let parsed = match parsed_result {
        Ok(parsed) => parsed,
        Err(ParseError::Empty) => return Ok(QueryOutcome::Empty),
        // LIMIT/OFFSET negativity is already a precise typed-parser diagnosis. It is not an
        // indication that this is richer SQL for libpg_query to lower, and sending it there loses
        // the PostgreSQL-compatible error in favor of the general lowerer's broader diagnostic.
        Err(error @ (ParseError::NegativeLimit | ParseError::NegativeOffset)) => {
            session.mark_transaction_failed();
            return Err(map_parse_error(error));
        }
        // The typed parser intentionally rejects joins, expressions, and other richer SELECT
        // syntax. The engine's libpg_query lowering is the one general GPU relational path for
        // those statements; an actual syntax error or non-SELECT still fails pre-effect there.
        Err(_) if gpu_db_sql::is_select_statement(sql) => {
            return submit_general_select_text_inner(shared, session, sql);
        }
        Err(error) => {
            session.mark_transaction_failed();
            return Err(map_parse_error(error));
        }
    };
    #[cfg(feature = "probe-timing")]
    let probe_insert = matches!(parsed.command(), Command::Insert(_));
    #[cfg(feature = "probe-timing")]
    if probe_insert {
        shared.engine.record_insert_probe_facade_parse_bind_nanos(
            probe_statement_started.elapsed().as_nanos() as u64,
        );
    }
    let result = submit_parsed(shared, session, parsed);
    #[cfg(feature = "probe-timing")]
    if probe_insert
        && matches!(
            &result,
            Ok(QueryOutcome::Command {
                tag: CommandTag::Insert,
                ..
            })
        )
    {
        shared
            .engine
            .record_insert_probe_successful_simple_text_service_nanos(
                probe_statement_started.elapsed().as_nanos() as u64,
            );
    }
    result
}
