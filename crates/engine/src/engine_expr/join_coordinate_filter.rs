//! Streaming join coordinate post-filters.
//!
//! This leaf owns streaming-only predicate and visibility filtering over accumulated device
//! coordinates. The main non-streaming coordinate executor retains its inline filter pipeline.

use super::join_source::JoinExecSide;
use crate::engine_expr_ir::ResidentExpr;
use crate::relational_model::RelationalTable;
use crate::{Engine, ExecuteError};
use gpu_db_types::EngineError;

impl Engine {
    pub(crate) fn filter_outer_join_projection_coordinates(
        &self,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        predicates: &[Option<ResidentExpr>],
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<gpu_db_execution::CudaJoinCoordinatesU32, ExecuteError> {
        if tables.len() != sides.len() || predicates.len() != sides.len() {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "outer-coordinate filter arity does not match the join".to_string(),
            )));
        }
        let mut masks = Vec::with_capacity(sides.len());
        let mut pad_masks = Vec::with_capacity(sides.len());
        for relation in 0..sides.len() {
            masks.push(self.resident_predicate_device_mask(
                predicates[relation].as_ref(),
                &tables[relation],
                &sides[relation].0.descriptor,
                sides[relation].1.mem(),
                sides[relation].2 as u32,
                sides[relation].3,
            )?);
            pad_masks.push(
                predicates[relation]
                    .as_ref()
                    .map(|predicate| self.predicate_mask_on_null_pad(predicate, &tables[relation]))
                    .transpose()?,
            );
        }
        sides[0]
            .1
            .mem()
            .filter_join_coordinates(
                coordinates,
                &masks.iter().map(Option::as_ref).collect::<Vec<_>>(),
                &pad_masks
                    .iter()
                    .map(|guard| guard.as_ref().and_then(|guard| guard.mask.as_ref()))
                    .collect::<Vec<_>>(),
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }

    pub(crate) fn filter_join_visibility_coordinates(
        &self,
        tables: &[RelationalTable],
        sides: &[JoinExecSide],
        coordinates: &gpu_db_execution::CudaJoinCoordinatesU32,
    ) -> Result<gpu_db_execution::CudaJoinCoordinatesU32, ExecuteError> {
        let mut masks = Vec::with_capacity(sides.len());
        for relation in 0..sides.len() {
            masks.push(self.resident_predicate_device_mask(
                None,
                &tables[relation],
                &sides[relation].0.descriptor,
                sides[relation].1.mem(),
                sides[relation].2 as u32,
                sides[relation].3,
            )?);
        }
        sides[0]
            .1
            .mem()
            .filter_join_coordinates(
                coordinates,
                &masks.iter().map(Option::as_ref).collect::<Vec<_>>(),
                &vec![None; sides.len()],
            )
            .map_err(|err| ExecuteError::Engine(EngineError::ApplyFailed(err.to_string())))
    }
}
