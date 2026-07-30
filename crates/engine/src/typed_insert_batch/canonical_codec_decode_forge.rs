//! Test-only rehashed-wire adversaries for the strict decoder.

use super::reencode;
use super::*;

#[cfg(test)]
#[derive(Clone, Copy)]
pub(in super::super) enum RehashedForgery {
    ReturningProjection,
    TargetIndex,
    ForeignKey,
    ForeignKeySupportingIndexMetadata,
    SelfForeignKeySupportingIndexIdentity,
    ExternalSupportingIndexConflict,
    SequenceDuplicateLocal,
    SequenceEffectiveIdentity,
    SequenceTransitionMatchesParent,
    SequenceTransitionReverse,
    SequenceMixedMode,
    SequenceInt8Target,
    SequenceAutocommitPrivate,
    SequencePrivateOwnerFuture,
    SequenceClassNameCollision,
    IndexClassNameCollision,
    SupportingIndexClassNameCollision,
    SupportingIndexPairClassNameCollision,
    IndexSequenceClassNameCollision,
    IndexOidOutOfRange,
    IndexOidTargetRelationCollision,
    IndexOidDependencyRelationCollision,
    IndexOidDomainCollision,
    SupportingIndexOidDomainCollision,
    IndexOidSequenceCollision,
    ExternalColumnIdCollision,
    UnindexedTargetExternalColumnIdCollision,
    ExternalColumnHugeOrdinal,
    ForeignKeyTypeOidRelationCollision,
    DomainOidNameCollision,
    TargetAttnumOrder,
    DateCarrierUnderflow,
    DateCarrierOverflow,
    TimestampCarrierUnderflow,
    TimestampCarrierOverflow,
}

/// Test-only adversarial wire constructor. It starts from a fully validated model, fabricates one
/// cross-section identity, recomputes every exposed digest, and serializes the forged model. The
/// public decoder must still reject it on structural metadata checks.
#[cfg(test)]
pub(in super::super) fn rehashed_forgery_for_test(
    bytes: &[u8],
    forgery: RehashedForgery,
) -> Result<Vec<u8>, EngineError> {
    let mut model = parse_valid_model(bytes)?;
    match forgery {
        RehashedForgery::ReturningProjection => {
            let projection = model
                .returning
                .projections
                .first_mut()
                .ok_or_else(|| codec_error("RETURNING fixture has no projection"))?;
            projection.column_id ^= 0x40;
            model.returning.digest = decoded_returning_digest(&model.returning)?;
            model.returning_digest = model.returning.digest;
        }
        RehashedForgery::TargetIndex => {
            let index = model
                .indexes
                .first_mut()
                .ok_or_else(|| codec_error("index fixture has no target index"))?;
            index.key_columns[0].column_id ^= 0x40;
        }
        RehashedForgery::ForeignKey => {
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?;
            foreign_key.parent_column.column_id ^= 0x40;
        }
        RehashedForgery::ForeignKeySupportingIndexMetadata => {
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?;
            foreign_key.supporting_index.key_columns[0].catalog_column_ordinal ^= 0x40;
        }
        RehashedForgery::SelfForeignKeySupportingIndexIdentity => {
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("self-referencing foreign-key fixture has no FK"))?;
            foreign_key.supporting_index.oid ^= 0x40;
        }
        RehashedForgery::ExternalSupportingIndexConflict => {
            let foreign_key = model
                .foreign_keys
                .get_mut(1)
                .ok_or_else(|| codec_error("external supporting-index fixture needs two FKs"))?;
            foreign_key.supporting_index.name =
                format!("{}_forged", foreign_key.supporting_index.name);
        }
        RehashedForgery::SequenceDuplicateLocal => {
            let first = model
                .effects
                .effects
                .first()
                .ok_or_else(|| codec_error("sequence fixture needs one effect"))?
                .request
                .expression_ordinal;
            let parent = model
                .effects
                .parent
                .ok_or_else(|| codec_error("sequence fixture lacks parent"))?;
            let effect = model
                .effects
                .effects
                .get_mut(1)
                .ok_or_else(|| codec_error("sequence fixture needs two effects"))?;
            effect.request.expression_ordinal = first;
            effect.request.absolute_expression_ordinal = parent
                .expression_base
                .checked_add(first)
                .ok_or_else(|| codec_error("forged sequence absolute overflows"))?;
        }
        RehashedForgery::SequenceEffectiveIdentity => {
            let effect = model
                .effects
                .effects
                .get_mut(1)
                .ok_or_else(|| codec_error("sequence fixture needs two effects"))?;
            effect.request.effective_name.push_str("_forged");
            effect.request.descriptor_digest = crate::sequence_descriptor_digest(
                effect.request.sequence_oid,
                &effect.request.effective_name,
            );
        }
        RehashedForgery::SequenceTransitionMatchesParent => {
            let parent = model
                .effects
                .parent
                .ok_or_else(|| codec_error("sequence fixture lacks parent"))?;
            let effect = model
                .effects
                .effects
                .first_mut()
                .ok_or_else(|| codec_error("sequence fixture needs one effect"))?;
            let DecodedEffectKind::Published {
                transition_txn_id, ..
            } = &mut effect.kind
            else {
                return Err(codec_error("sequence fixture effect is not published"));
            };
            *transition_txn_id = parent.txn_id;
        }
        RehashedForgery::SequenceTransitionReverse => {
            let effects = &mut model.effects.effects;
            if effects.len() < 2 {
                return Err(codec_error("sequence fixture needs two effects"));
            }
            let (left, right) = effects.split_at_mut(1);
            let DecodedEffectKind::Published {
                transition_txn_id: first,
                ..
            } = &mut left[0].kind
            else {
                return Err(codec_error("first sequence effect is not published"));
            };
            let DecodedEffectKind::Published {
                transition_txn_id: second,
                ..
            } = &mut right[0].kind
            else {
                return Err(codec_error("second sequence effect is not published"));
            };
            std::mem::swap(first, second);
        }
        RehashedForgery::SequenceMixedMode => {
            let parent = model
                .effects
                .parent
                .ok_or_else(|| codec_error("sequence fixture lacks parent"))?;
            let effect = model
                .effects
                .effects
                .get_mut(1)
                .ok_or_else(|| codec_error("private sequence fixture needs two effects"))?;
            let input_digest = sequence_input_digest(
                parent.view(),
                effect.request.view(),
                effect.request.absolute_expression_ordinal,
            );
            effect.kind = DecodedEffectKind::Published {
                transition_txn_id: 1,
                input_digest,
                returned_value: effect.request.value,
            };
        }
        RehashedForgery::SequenceAutocommitPrivate => {
            forge_autocommit_private(&mut model)?;
        }
        RehashedForgery::SequencePrivateOwnerFuture => {
            forge_private_owner_ordinal(&mut model, true)?;
        }
        RehashedForgery::SequenceInt8Target => {
            let request = model
                .effects
                .effects
                .first()
                .ok_or_else(|| codec_error("sequence fixture needs one effect"))?
                .request
                .clone();
            let column = model
                .columns
                .get_mut(request.catalog_column_ordinal as usize)
                .ok_or_else(|| codec_error("sequence fixture column is absent"))?;
            column.ty = SqlType::Int8;
            column.type_oid = SqlType::Int8.postgres_oid();
            column.type_size = SqlType::Int8.type_size();
            column.values = TypedInsertColumnValues::I64(vec![request.value].into());
        }
        RehashedForgery::SequenceClassNameCollision => {
            let effect = model
                .effects
                .effects
                .first_mut()
                .ok_or_else(|| codec_error("sequence fixture needs one effect"))?;
            effect.request.effective_name = model.target.name.clone();
            effect.request.descriptor_digest = crate::sequence_descriptor_digest(
                effect.request.sequence_oid,
                &effect.request.effective_name,
            );
        }
        RehashedForgery::IndexClassNameCollision => {
            let index = model
                .indexes
                .first_mut()
                .ok_or_else(|| codec_error("index fixture has no target index"))?;
            index.name = model.target.name.clone();
        }
        RehashedForgery::SupportingIndexClassNameCollision => {
            let name = model
                .indexes
                .first()
                .ok_or_else(|| codec_error("index fixture has no target index"))?
                .name
                .clone();
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?;
            foreign_key.supporting_index.name = name;
        }
        RehashedForgery::SupportingIndexPairClassNameCollision => {
            let name = model
                .foreign_keys
                .first()
                .ok_or_else(|| codec_error("foreign-key fixture has no first FK"))?
                .supporting_index
                .name
                .clone();
            let foreign_key = model
                .foreign_keys
                .get_mut(1)
                .ok_or_else(|| codec_error("foreign-key fixture needs two FKs"))?;
            foreign_key.supporting_index.name = name;
        }
        RehashedForgery::IndexSequenceClassNameCollision => {
            let name = model
                .effects
                .effects
                .first()
                .ok_or_else(|| codec_error("sequence fixture needs one effect"))?
                .request
                .effective_name
                .clone();
            let index = model
                .indexes
                .first_mut()
                .ok_or_else(|| codec_error("index fixture has no target index"))?;
            index.name = name;
        }
        RehashedForgery::IndexOidOutOfRange => {
            let index = model
                .indexes
                .first_mut()
                .ok_or_else(|| codec_error("index fixture has no target index"))?;
            index.oid = i32::MAX as u32 + 1;
        }
        RehashedForgery::IndexOidTargetRelationCollision => {
            let index = model
                .indexes
                .first_mut()
                .ok_or_else(|| codec_error("index fixture has no target index"))?;
            index.oid = model.target.oid;
        }
        RehashedForgery::IndexOidDependencyRelationCollision => {
            let oid = model
                .dependencies
                .get(1)
                .ok_or_else(|| codec_error("index fixture needs an external dependency"))?
                .oid;
            let index = model
                .indexes
                .first_mut()
                .ok_or_else(|| codec_error("index fixture has no target index"))?;
            index.oid = oid;
        }
        RehashedForgery::IndexOidDomainCollision => {
            let oid = model
                .domains
                .first()
                .ok_or_else(|| codec_error("index fixture needs a domain"))?
                .oid;
            let index = model
                .indexes
                .first_mut()
                .ok_or_else(|| codec_error("index fixture has no target index"))?;
            index.oid = oid;
        }
        RehashedForgery::SupportingIndexOidDomainCollision => {
            let oid = model
                .domains
                .first()
                .ok_or_else(|| codec_error("foreign-key fixture needs a domain"))?
                .oid;
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?;
            foreign_key.supporting_index.oid = oid;
        }
        RehashedForgery::IndexOidSequenceCollision => {
            let oid = model
                .effects
                .effects
                .first()
                .ok_or_else(|| codec_error("sequence fixture needs one effect"))?
                .request
                .sequence_oid;
            let index = model
                .indexes
                .first_mut()
                .ok_or_else(|| codec_error("index fixture has no target index"))?;
            index.oid = oid;
        }
        RehashedForgery::ExternalColumnIdCollision => {
            let child_id = model
                .foreign_keys
                .first()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?
                .child_column
                .column_id;
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?;
            foreign_key.parent_column.column_id = child_id;
            foreign_key.supporting_index.key_columns[0].column_id = child_id;
        }
        RehashedForgery::UnindexedTargetExternalColumnIdCollision => {
            let column_id = model
                .columns
                .first()
                .ok_or_else(|| codec_error("fixture has no target column"))?
                .column_id;
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?;
            foreign_key.parent_column.column_id = column_id;
            foreign_key.supporting_index.key_columns[0].column_id = column_id;
        }
        RehashedForgery::ExternalColumnHugeOrdinal => {
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?;
            foreign_key.parent_column.catalog_column_ordinal = u32::MAX;
            foreign_key.supporting_index.key_columns[0].catalog_column_ordinal = u32::MAX;
        }
        RehashedForgery::ForeignKeyTypeOidRelationCollision => {
            let foreign_key = model
                .foreign_keys
                .first_mut()
                .ok_or_else(|| codec_error("foreign-key fixture has no FK"))?;
            foreign_key.parent_column.type_oid = model.target.oid;
            foreign_key.supporting_index.key_columns[0].type_oid = model.target.oid;
        }
        RehashedForgery::DomainOidNameCollision => {
            let domain = model
                .domains
                .first()
                .ok_or_else(|| codec_error("domain fixture has no domain"))?;
            model.domains.push(DecodedDomain {
                schema: domain.schema.clone(),
                name: format!("{}_forged", domain.name),
                oid: domain.oid,
                base_type: domain.base_type,
            });
        }
        RehashedForgery::TargetAttnumOrder => {
            let column = model
                .columns
                .first_mut()
                .ok_or_else(|| codec_error("fixture has no target column"))?;
            column.attnum = 3;
        }
        RehashedForgery::DateCarrierUnderflow => {
            let column = model
                .columns
                .first_mut()
                .ok_or_else(|| codec_error("date fixture has no column"))?;
            column.values = TypedInsertColumnValues::I32(
                vec![gpu_db_sql::datetime::PG_DATE_MIN_DAYS - 1].into(),
            );
        }
        RehashedForgery::DateCarrierOverflow => {
            let column = model
                .columns
                .first_mut()
                .ok_or_else(|| codec_error("date fixture has no column"))?;
            column.values = TypedInsertColumnValues::I32(
                vec![gpu_db_sql::datetime::PG_DATE_END_DAYS_EXCLUSIVE].into(),
            );
        }
        RehashedForgery::TimestampCarrierUnderflow => {
            let column = model
                .columns
                .first_mut()
                .ok_or_else(|| codec_error("timestamp fixture has no column"))?;
            column.values = TypedInsertColumnValues::I64(
                vec![gpu_db_sql::datetime::PG_TIMESTAMP_MIN_MICROS - 1].into(),
            );
        }
        RehashedForgery::TimestampCarrierOverflow => {
            let column = model
                .columns
                .first_mut()
                .ok_or_else(|| codec_error("timestamp fixture has no column"))?;
            column.values = TypedInsertColumnValues::I64(
                vec![gpu_db_sql::datetime::PG_TIMESTAMP_END_MICROS_EXCLUSIVE].into(),
            );
        }
    }
    rehash_model_and_sequence_witnesses(&mut model)?;
    if matches!(
        forgery,
        RehashedForgery::DateCarrierUnderflow
            | RehashedForgery::DateCarrierOverflow
            | RehashedForgery::TimestampCarrierUnderflow
            | RehashedForgery::TimestampCarrierOverflow
    ) {
        reencode::reencode_decoded_unchecked(&model)
    } else {
        reencode::reencode_decoded(&model)
    }
}

fn forge_private_owner_ordinal(model: &mut DecodedModel, future: bool) -> Result<(), EngineError> {
    let parent = model
        .effects
        .parent
        .ok_or_else(|| codec_error("private sequence fixture lacks parent"))?;
    let first = model
        .effects
        .effects
        .first()
        .cloned()
        .ok_or_else(|| codec_error("private sequence fixture needs two effects"))?;
    let DecodedEffectKind::Private {
        prior_last_value,
        prior_is_called,
        next_last_value,
        next_is_called,
        lifetime_origin,
        mut owner,
        input_digest,
        ..
    } = first.kind
    else {
        return Err(codec_error("first sequence effect is not private"));
    };
    owner.statement_ordinal = parent.statement_ordinal.as_u32() + u32::from(future);
    let predecessor = PrivatePredecessor::Lifecycle(owner);
    let child = private_child_digest(
        parent.view(),
        first.request.view(),
        first.request.absolute_expression_ordinal,
        input_digest,
        first.request.descriptor_digest,
        lifetime_origin,
        (prior_last_value, prior_is_called),
        predecessor,
    )?;
    let outcome = private_outcome_digest(
        child,
        owner,
        first.request.value,
        (next_last_value, next_is_called),
    )?;
    let first_effect = model
        .effects
        .effects
        .first_mut()
        .ok_or_else(|| codec_error("private sequence fixture needs first effect"))?;
    let DecodedEffectKind::Private {
        owner: first_owner,
        predecessor: first_predecessor,
        ..
    } = &mut first_effect.kind
    else {
        return Err(codec_error("first sequence effect is not private"));
    };
    *first_owner = owner;
    *first_predecessor = predecessor;
    let second = model
        .effects
        .effects
        .get_mut(1)
        .ok_or_else(|| codec_error("private sequence fixture needs second effect"))?;
    let DecodedEffectKind::Private {
        owner: second_owner,
        predecessor: second_predecessor,
        ..
    } = &mut second.kind
    else {
        return Err(codec_error("second sequence effect is not private"));
    };
    *second_owner = owner;
    *second_predecessor = PrivatePredecessor::Outcome(outcome);
    Ok(())
}

fn forge_autocommit_private(model: &mut DecodedModel) -> Result<(), EngineError> {
    model.target.statement_ordinal = InsertStatementOrdinal::FIRST;
    for effect in &mut model.effects.effects {
        effect.request.statement_ordinal = InsertStatementOrdinal::FIRST;
        effect.request.absolute_expression_ordinal = effect.request.expression_ordinal;
    }
    let parent = model
        .effects
        .parent
        .as_mut()
        .ok_or_else(|| codec_error("private sequence fixture lacks parent"))?;
    parent.autocommit = true;
    parent.statement_ordinal = InsertStatementOrdinal::FIRST;
    parent.expression_base = 0;
    for effect in &mut model.effects.effects {
        let DecodedEffectKind::Private { owner, .. } = &mut effect.kind else {
            return Err(codec_error("autocommit fixture effect is not private"));
        };
        owner.statement_ordinal = InsertStatementOrdinal::FIRST.as_u32();
    }
    Ok(())
}

fn rehash_model_and_sequence_witnesses(model: &mut DecodedModel) -> Result<(), EngineError> {
    model.typed_statement_digest = decoded_statement_digest(
        &model.target,
        &model.columns,
        &model.dependencies,
        &model.domains,
        &model.indexes,
        &model.foreign_keys,
        model.returning.digest,
        &model.effects,
    )?;
    let Some(parent) = model.effects.parent.as_mut() else {
        return Ok(());
    };
    parent.request_digest = model.typed_statement_digest;
    refresh_sequence_witnesses(model)
}

fn refresh_sequence_witnesses(model: &mut DecodedModel) -> Result<(), EngineError> {
    let parent = model
        .effects
        .parent
        .ok_or_else(|| codec_error("private sequence fixture lacks parent"))?;
    let mut private_outcomes = BTreeMap::new();
    for ordinal in 0..model.effects.effects.len() {
        let effect = model.effects.effects[ordinal].clone();
        let input_digest = sequence_input_digest(
            parent.view(),
            effect.request.view(),
            effect.request.absolute_expression_ordinal,
        );
        match effect.kind {
            DecodedEffectKind::Published {
                transition_txn_id,
                returned_value,
                ..
            } => {
                model.effects.effects[ordinal].kind = DecodedEffectKind::Published {
                    transition_txn_id,
                    input_digest,
                    returned_value,
                };
            }
            DecodedEffectKind::Private {
                prior_last_value,
                prior_is_called,
                next_last_value,
                next_is_called,
                lifetime_origin,
                owner,
                ..
            } => {
                let predecessor = private_outcomes
                    .get(&effect.request.sequence_oid)
                    .copied()
                    .map(PrivatePredecessor::Outcome)
                    .unwrap_or(PrivatePredecessor::Lifecycle(owner));
                let child = private_child_digest(
                    parent.view(),
                    effect.request.view(),
                    effect.request.absolute_expression_ordinal,
                    input_digest,
                    effect.request.descriptor_digest,
                    lifetime_origin,
                    (prior_last_value, prior_is_called),
                    predecessor,
                )?;
                let outcome = private_outcome_digest(
                    child,
                    owner,
                    effect.request.value,
                    (next_last_value, next_is_called),
                )?;
                model.effects.effects[ordinal].kind = DecodedEffectKind::Private {
                    prior_last_value,
                    prior_is_called,
                    next_last_value,
                    next_is_called,
                    lifetime_origin,
                    owner,
                    predecessor,
                    input_digest,
                    chain: PrivateChain {
                        state: (next_last_value, next_is_called),
                        owner,
                        outcome,
                    },
                };
                private_outcomes.insert(effect.request.sequence_oid, outcome);
            }
        }
    }
    Ok(())
}
