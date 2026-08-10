//! Source-backed S2 record fill.
//!
//! Each record is remeasured through a bounded aggregate-region reader before one exact local
//! copy is made.  The copy is immediately decoded into its move-only owner and then dropped;
//! neither this module nor the retained graph keeps a raw S2 byte carrier.

use super::fixed::push_exact;
use super::source::{exact_copy_scratch, fill_error, AggregateRegionSource};
use super::ObservedSourceMeasure;
use crate::typed_insert_aggregate::codec::DecodedAggregateFraming;
use crate::typed_insert_aggregate::semantics_v2::retained::graph::ReservedSemanticsV2Graph;
use crate::typed_insert_batch::{
    copy_decoded_canonical_typed_insert_after_measure,
    decode_decoded_canonical_typed_insert_after_measure,
    measure_decoded_canonical_typed_insert_from_source,
};
use crate::EngineError;
use sha2::{Digest, Sha256};

pub(super) fn fill_s2_records(
    framing: &DecodedAggregateFraming<'_>,
    graph: &mut ReservedSemanticsV2Graph,
    observed: &mut ObservedSourceMeasure,
) -> Result<(), EngineError> {
    framing.with_section_reader(1, |reader| {
        let mut record_start = 0_u64;
        while !reader.done() {
            let record_bytes = reader.u32()?;
            let bytes = u64::from(record_bytes);
            if bytes == 0 || bytes > reader.remaining() {
                return Err(fill_error("S2 retained record range is invalid"));
            }
            let source = AggregateRegionSource::new(
                framing,
                1,
                record_start
                    .checked_add(4)
                    .ok_or_else(|| fill_error("S2 retained record offset overflows"))?,
                bytes,
            );
            let measure = measure_decoded_canonical_typed_insert_from_source(&source)
                .map_err(|_| fill_error("S2 retained source measurement fails strict decode"))?;
            if measure.record_bytes() != bytes {
                return Err(fill_error("S2 retained source measure length drifted"));
            }
            let scratch_bytes = measure
                .maximum_with_record_copy_bytes()
                .map_err(|_| fill_error("S2 retained copy scratch measure overflows"))?;
            let scratch_slots = measure
                .maximum_with_record_copy_allocation_slots()
                .map_err(|_| fill_error("S2 retained copy scratch slot measure overflows"))?;
            observed.checked_add_s2(
                measure.persistent_bytes(),
                measure.persistent_allocation_slots(),
                scratch_bytes,
                scratch_slots,
            )?;

            let mut scratch = exact_copy_scratch(bytes, "S2 exact source copy")?;
            copy_decoded_canonical_typed_insert_after_measure(&source, measure, &mut scratch)
                .map_err(|_| fill_error("S2 retained source changed after its measurement"))?;
            let record_digest = s2_record_digest(record_bytes, &scratch);
            let decoded = decode_decoded_canonical_typed_insert_after_measure(&scratch, measure)
                .map_err(|_| fill_error("S2 retained exact copy fails strict decode"))?;
            drop(scratch);

            let ordinal = graph.records.len();
            let statement = graph
                .statements
                .get_mut(ordinal)
                .ok_or_else(|| fill_error("S2 retained record has no reserved S1 statement"))?;
            let facts = decoded.facts();
            if statement.statement_ordinal != facts.statement_ordinal.as_u32()
                || statement.input_row_count != facts.row_count
                || statement.typed_statement_digest != facts.typed_statement_digest
            {
                return Err(fill_error(
                    "S2 decoded record does not match its S1 statement",
                ));
            }
            statement.record_bytes = record_bytes;
            statement.record_digest = record_digest;
            push_exact(
                &mut graph.records,
                decoded,
                "S2 decoded typed-record directory",
            )?;

            reader.skip(bytes)?;
            record_start = record_start
                .checked_add(4)
                .and_then(|value| value.checked_add(bytes))
                .ok_or_else(|| fill_error("S2 retained record cursor overflows"))?;
        }
        Ok(())
    })
}

fn s2_record_digest(record_bytes: u32, bytes: &[u8]) -> [u8; 32] {
    let domain = b"gpu-db/write001/s7-s2-record/v2";
    let mut digest = Sha256::new();
    digest.update((domain.len() as u64).to_le_bytes());
    digest.update(domain);
    digest.update(record_bytes.to_le_bytes());
    digest.update(bytes);
    digest.finalize().into()
}
