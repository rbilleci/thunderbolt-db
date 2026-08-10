//! Checked row-identity range shared by codec-5 live apply and recovery.
//!
//! The displaced v1 fixed-INSERT template/bound carrier was deleted with the direct commit-wave
//! encoder. This type is not a WAL carrier: it only keeps allocator bounds and the exact device
//! row identities tied to one checked range.

use super::*;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ProposedRowIdRange {
    first: u64,
    count: u32,
    allocator_high_water: u64,
}

impl ProposedRowIdRange {
    pub(crate) fn new(first: u64, count: u32) -> Result<Self, EngineError> {
        if first == 0 || count == 0 {
            return Err(EngineError::Durability(
                "proposed INSERT row-id range must be nonzero".to_string(),
            ));
        }
        let allocator_high_water = first.checked_add(u64::from(count)).ok_or_else(|| {
            EngineError::Durability("proposed INSERT row-id range overflows".to_string())
        })?;
        Ok(Self {
            first,
            count,
            allocator_high_water,
        })
    }

    #[cfg(test)]
    fn row_id_at(&self, offset: usize) -> Result<u64, EngineError> {
        let offset = u64::try_from(offset).map_err(|_| {
            EngineError::Durability("proposed INSERT row-id offset overflows".to_string())
        })?;
        self.first.checked_add(offset).ok_or_else(|| {
            EngineError::Durability("proposed INSERT row-id range overflows".to_string())
        })
    }

    #[cfg(test)]
    pub(crate) const fn count(&self) -> u32 {
        self.count
    }

    pub(crate) fn first(&self) -> u64 {
        self.first
    }

    pub(crate) fn allocator_high_water(&self) -> u64 {
        self.allocator_high_water
    }

    #[cfg(test)]
    pub(crate) fn exact_row_ids(&self) -> Result<Box<[u64]>, EngineError> {
        (0..self.count as usize)
            .map(|offset| self.row_id_at(offset))
            .collect::<Result<Vec<_>, _>>()
            .map(Vec::into_boxed_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposed_range_is_nonzero_bounded_and_exact() {
        assert!(ProposedRowIdRange::new(0, 1).is_err());
        assert!(ProposedRowIdRange::new(1, 0).is_err());
        assert!(ProposedRowIdRange::new(u64::MAX, 1).is_err());
        let proposed = ProposedRowIdRange::new(41, 3).unwrap();
        assert_eq!(proposed.first(), 41);
        assert_eq!(proposed.count(), 3);
        assert_eq!(proposed.allocator_high_water(), 44);
        assert_eq!(&*proposed.exact_row_ids().unwrap(), &[41, 42, 43]);
    }
}
