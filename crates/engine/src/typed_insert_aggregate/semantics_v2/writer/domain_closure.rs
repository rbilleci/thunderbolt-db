//! Static domain dependency closure for the canonical INSERT writer.
//!
//! Domain values already use their base-type device vectors. This leaf only binds every sealed
//! S2 domain source to the existing S7 kind/role-7 grammar; it owns no cast, constraint verdict,
//! catalog mutation, WAL lifecycle, or apply path.

use super::*;

pub(super) struct DomainClosure {
    pub(super) dependencies: Vec<[u8; 224]>,
    pub(super) dependency_uses: Vec<[u8; 32]>,
}

pub(super) fn from_sealed_records(
    input: &LiveTypedInsertView<'_>,
    records: &[DecodedTypedInsertRecord],
    target_table_ref: u32,
    first_dependency_ref: u32,
) -> Result<DomainClosure, EngineError> {
    let mut identities = input
        .statements
        .iter()
        .map(|statement| {
            records
                .get(statement.statement_ordinal as usize)
                .ok_or_else(|| error("domain S7 statement record is absent"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flat_map(|record| record.domains())
        .map(|domain| {
            (
                domain.oid,
                domain.schema.to_owned(),
                domain.name.to_owned(),
                domain.base_type,
            )
        })
        .collect::<Vec<_>>();
    identities.sort_unstable_by(|left, right| {
        (left.0, left.1.as_str(), left.2.as_str()).cmp(&(
            right.0,
            right.1.as_str(),
            right.2.as_str(),
        ))
    });
    identities.dedup();
    if identities.windows(2).any(|pair| {
        pair[0].0 == pair[1].0
            || (pair[0].1.as_str(), pair[0].2.as_str()) == (pair[1].1.as_str(), pair[1].2.as_str())
    }) {
        return Err(error(
            "sealed S2 domains do not have unique catalog identities",
        ));
    }

    let mut dependencies = Vec::with_capacity(identities.len());
    for (oid, schema, name, base_type) in &identities {
        let dependency_ref = first_dependency_ref
            .checked_add(
                u32::try_from(dependencies.len())
                    .map_err(|_| error("S7 domain dependency count exceeds u32"))?,
            )
            .ok_or_else(|| error("S7 domain dependency reference overflows"))?;
        dependencies.push(domain_dependency(
            input,
            dependency_ref,
            target_table_ref,
            *oid,
            schema,
            name,
            *base_type,
        )?);
    }

    let mut dependency_uses = Vec::new();
    for statement in input.statements {
        let statement_ordinal = statement.statement_ordinal;
        let record = records
            .get(statement_ordinal as usize)
            .ok_or_else(|| error("domain S7 statement record is absent"))?;
        for source in record.domains() {
            let identity_ordinal = identities
                .iter()
                .position(|identity| {
                    identity.0 == source.oid
                        && identity.1 == source.schema
                        && identity.2 == source.name
                        && identity.3 == source.base_type
                })
                .ok_or_else(|| error("S2 domain source has no S7 dependency token"))?;
            let dependency_ref = first_dependency_ref
                .checked_add(
                    u32::try_from(identity_ordinal)
                        .map_err(|_| error("S7 domain dependency ordinal exceeds u32"))?,
                )
                .ok_or_else(|| error("S7 domain dependency reference overflows"))?;
            dependency_uses.push(dependency_use(
                statement_ordinal,
                dependency_ref,
                7,
                source.ordinal,
                ABSENT_U32,
                ABSENT_U32,
            ));
        }
    }
    Ok(DomainClosure {
        dependencies,
        dependency_uses,
    })
}

fn domain_dependency(
    input: &LiveTypedInsertView<'_>,
    reference: u32,
    target_table_ref: u32,
    oid: u32,
    schema: &str,
    name: &str,
    base_type: crate::SqlType,
) -> Result<[u8; 224], EngineError> {
    if oid == 0 || oid > 0x7fff_ffff || schema.is_empty() || name.is_empty() {
        return Err(error("sealed S2 domain identity is invalid"));
    }
    let stable_domain_id = u64::from(oid);
    let shape_digest = v2_digest(
        b"gpu-db/write001/s7-domain-shape/v2",
        &[
            &crate::typed_insert_batch::typed_image_sql_storage(base_type),
            &base_type.postgres_oid().to_le_bytes(),
            &base_type.type_size().to_le_bytes(),
        ],
    );
    let name_digest = qualified_name_digest(schema, name)?;
    let identity = v2_digest(
        b"gpu-db/write001/s7-domain-object/v2",
        &[
            &stable_domain_id.to_le_bytes(),
            &oid.to_le_bytes(),
            &input.identity.catalog_epoch.to_le_bytes(),
            &input.identity.catalog_epoch.to_le_bytes(),
            &shape_digest,
            &name_digest,
        ],
    );
    let mut raw = [0_u8; 224];
    put_u32(&mut raw, 0, reference);
    raw[4] = 7;
    raw[5] = 1;
    put_u64(&mut raw, 8, stable_domain_id);
    put_u32(&mut raw, 16, oid);
    put_u32(&mut raw, 20, target_table_ref);
    put_u64(&mut raw, 24, input.identity.catalog_epoch);
    put_u64(&mut raw, 32, input.identity.dependency_validation_floor);
    put_u32(&mut raw, 40, ABSENT_U32);
    put_u32(&mut raw, 44, ABSENT_U32);
    put_u64(&mut raw, 48, input.identity.catalog_epoch);
    put_digest(&mut raw, 64, shape_digest);
    put_digest(&mut raw, 128, name_digest);
    put_digest(&mut raw, 160, identity);
    let token_digest = v2_digest(
        b"gpu-db/write001/s7-dependency-token/v2",
        &[&raw[..192], &[0; 32], &[0; 32], &[0; 32]],
    );
    put_digest(&mut raw, 192, token_digest);
    Ok(raw)
}
