//! Non-table object DDL (P0 §9.6 decomposition, behavior-preserving): a focused
//! `impl Engine` block for CREATE/RENAME of views and CREATE/DROP/RENAME of
//! materialized views (with dependency tracking and refresh), functions,
//! sequences (implicit sequences, nextval/setval, column-default evaluation,
//! and DROP), domains, schemas, databases, and tablespaces.

use super::*;

impl Engine {
    pub(crate) fn apply_create_view(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateView,
    ) -> Result<(), EngineError> {
        self.apply_create_view_with_replay_policy(cat, create, false)
    }

    pub(crate) fn apply_create_view_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateView,
    ) -> Result<(), EngineError> {
        self.apply_create_view_with_replay_policy(cat, create, true)
    }

    fn apply_create_view_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let target_conflicts = if legacy_replay {
            cat.relational_catalog.contains_key(&create.name)
                || cat.relational_materialized_views.contains_key(&create.name)
                || cat.relational_sequences.contains_key(&create.name)
                || (!create.or_replace && cat.relational_views.contains_key(&create.name))
        } else {
            let target_kind = cat.pg_class_relation_kind(&create.name)?;
            target_kind.is_some()
                && !(create.or_replace && target_kind == Some(PgClassRelationKind::View))
        };
        if target_conflicts {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.name
            )));
        }
        if create.or_replace && self.relational_view_has_dependents(&create.name) {
            return Err(EngineError::ApplyFailed(
                "cannot replace view because another view depends on it".to_string(),
            ));
        }
        if legacy_replay {
            if cat
                .relational_materialized_views
                .contains_key(&create.query.table)
            {
                return Err(EngineError::ApplyFailed(
                    "views over materialized views are unsupported".to_string(),
                ));
            }
            if cat.relational_views.contains_key(&create.query.table) {
                if self.relational_view_depends_on(&create.query.table, &create.name) {
                    return Err(EngineError::ApplyFailed(
                        "view dependency cycle is unsupported".to_string(),
                    ));
                }
            } else if !cat.relational_catalog.contains_key(&create.query.table) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    create.query.table
                )));
            }
        } else {
            match cat.pg_class_relation_kind(&create.query.table)? {
                Some(PgClassRelationKind::Table) => {}
                Some(PgClassRelationKind::View) => {
                    if self.relational_view_depends_on(&create.query.table, &create.name) {
                        return Err(EngineError::ApplyFailed(
                            "view dependency cycle is unsupported".to_string(),
                        ));
                    }
                }
                Some(PgClassRelationKind::MaterializedView) => {
                    return Err(EngineError::ApplyFailed(
                        "views over materialized views are unsupported".to_string(),
                    ))
                }
                Some(_) => {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" cannot be a stored-view dependency",
                        create.query.table
                    )))
                }
                None => {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" does not exist",
                        create.query.table
                    )))
                }
            }
        }
        let mut next = cat.clone();
        let (oid, acl) = if let Some(existing) = next.relational_views.get(&create.name) {
            (existing.oid, existing.acl.clone())
        } else if legacy_replay {
            let oid = next.relational_next_oid;
            next.relational_next_oid =
                next.relational_next_oid.checked_add(1).ok_or_else(|| {
                    EngineError::ApplyFailed("relational view OID allocation exhausted".to_string())
                })?;
            (oid, BTreeMap::new())
        } else {
            next.finalize_legacy_index_oid_migration()?;
            let oid =
                next.allocate_relational_class_oid("relational view OID allocation exhausted")?;
            (oid, BTreeMap::new())
        };
        next.relational_views.insert(
            create.name.clone(),
            RelationalView {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                query: create.query,
                definition: create.definition,
                acl,
            },
        );
        *cat = next;
        Ok(())
    }

    pub(crate) fn relational_view_depends_on(&self, view: &str, target: &str) -> bool {
        let mut seen = BTreeSet::new();
        self.relational_view_depends_on_inner(view, target, &mut seen)
    }

    fn relational_view_depends_on_inner(
        &self,
        view: &str,
        target: &str,
        seen: &mut BTreeSet<String>,
    ) -> bool {
        let cat = self.catalog_snapshot();
        if view == target {
            return true;
        }
        if !seen.insert(view.to_string()) {
            return false;
        }
        let Some(view) = cat.relational_views.get(view) else {
            return false;
        };
        self.relational_view_depends_on_inner(&view.query.table, target, seen)
    }

    pub(crate) fn relational_view_has_dependents(&self, view: &str) -> bool {
        let cat = self.catalog_snapshot();
        cat.relational_views.iter().any(|(candidate, _)| {
            candidate != view && self.relational_view_depends_on(candidate, view)
        })
    }

    pub(crate) fn apply_create_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateMaterializedView,
    ) -> Result<(), EngineError> {
        self.apply_create_materialized_view_with_replay_policy(cat, create, false)
    }

    pub(crate) fn apply_create_materialized_view_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateMaterializedView,
    ) -> Result<(), EngineError> {
        self.apply_create_materialized_view_with_replay_policy(cat, create, true)
    }

    fn apply_create_materialized_view_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateMaterializedView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        if legacy_replay {
            self.preflight_create_materialized_view_legacy_replay(&create)?;
        } else {
            self.preflight_create_materialized_view(&create)?;
        }
        // This SELECT runs INSIDE the commit critical section (the commit_mutex is held); suppress the
        // deep read executor's leader re-check on this thread so it does not self-deadlock re-locking it.
        let result = self
            .skip_leader_check_during_internal_read(|engine| {
                engine.execute_relational_select(&create.query)
            })
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        let mut next = cat.clone();
        let oid = if legacy_replay {
            let oid = next.relational_next_oid;
            next.relational_next_oid =
                next.relational_next_oid.checked_add(1).ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "relational materialized view OID allocation exhausted".to_string(),
                    )
                })?;
            oid
        } else {
            next.finalize_legacy_index_oid_migration()?;
            next.allocate_relational_class_oid(
                "relational materialized view OID allocation exhausted",
            )?
        };
        let mut columns = Vec::with_capacity(result.columns.len());
        let mut next_column_id = next.relational_next_column_id;
        for (idx, column) in result.columns.iter().cloned().enumerate() {
            let attnum = i16::try_from(idx + 1).map_err(|_| {
                EngineError::ApplyFailed(
                    "too many columns for bootstrap materialized view".to_string(),
                )
            })?;
            let id = next_column_id;
            next_column_id = next_column_id.checked_add(1).ok_or_else(|| {
                EngineError::ApplyFailed(
                    "relational materialized view column id allocation exhausted".to_string(),
                )
            })?;
            columns.push(RelationalColumn {
                id,
                table_oid: oid,
                attnum,
                name: column.name,
                ty: column.ty,
                domain: None,
                default: None,
                type_oid: column.type_oid,
                type_size: column.type_size,
            });
        }
        next.relational_next_column_id = next_column_id;
        let rows = if create.with_data {
            result.rows.into_boxed()
        } else {
            Vec::new()
        };
        next.relational_materialized_views.insert(
            create.name.clone(),
            RelationalMaterializedView {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                query: create.query,
                definition: create.definition,
                columns,
                rows,
                acl: BTreeMap::new(),
            },
        );
        *cat = next;
        Ok(())
    }

    pub(crate) fn apply_refresh_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        refresh: RefreshMaterializedView,
    ) -> Result<(), EngineError> {
        self.apply_refresh_materialized_view_with_replay_policy(cat, refresh, false)
    }

    pub(crate) fn apply_refresh_materialized_view_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        refresh: RefreshMaterializedView,
    ) -> Result<(), EngineError> {
        self.apply_refresh_materialized_view_with_replay_policy(cat, refresh, true)
    }

    fn apply_refresh_materialized_view_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        refresh: RefreshMaterializedView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        if legacy_replay {
            self.preflight_refresh_materialized_view_legacy_replay(&refresh)?;
        } else {
            self.preflight_refresh_materialized_view(&refresh)?;
        }
        let existing = cat
            .relational_materialized_views
            .get(&refresh.name)
            .ok_or_else(|| {
                EngineError::ApplyFailed(format!(
                    "materialized view \"{}\" does not exist",
                    refresh.name
                ))
            })?;
        let query = existing.query.clone();
        let columns = existing.columns.clone();
        // Mid-commit read (commit_mutex held): suppress the read executor's leader re-check.
        let result = self
            .skip_leader_check_during_internal_read(|engine| {
                engine.execute_relational_select(&query)
            })
            .map_err(|err| EngineError::ApplyFailed(err.to_string()))?;
        if result.columns.len() != columns.len()
            || result
                .columns
                .iter()
                .zip(columns.iter())
                .any(|(left, right)| left.name != right.name || left.ty != right.ty)
        {
            return Err(EngineError::ApplyFailed(
                "materialized view refresh changed the result shape".to_string(),
            ));
        }
        let view = cat
            .relational_materialized_views
            .get_mut(&refresh.name)
            .expect("materialized view existence preflighted");
        view.rows = result.rows.into_boxed();
        Ok(())
    }

    pub(crate) fn apply_create_function(
        &self,
        cat: &mut DdlCatalogState,
        create: gpu_db_sql::CreateFunction,
    ) -> Result<(), EngineError> {
        if cat.relational_functions.contains_key(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "function \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational function OID allocation exhausted".to_string())
        })?;
        cat.relational_functions.insert(
            create.name.clone(),
            RelationalFunction {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                return_type: create.return_type,
                body: create.body,
                acl: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub(crate) fn apply_drop_function(
        &self,
        cat: &mut DdlCatalogState,
        drop: gpu_db_sql::DropFunction,
    ) -> Result<(), EngineError> {
        if !drop.if_exists && !cat.relational_functions.contains_key(&drop.name) {
            return Err(EngineError::ApplyFailed(format!(
                "function \"{}\" does not exist",
                drop.name
            )));
        }
        cat.relational_functions.remove(&drop.name);
        cat.relational_comments
            .remove(&RelationalCommentTarget::Function {
                function: drop.name,
            });
        Ok(())
    }

    pub(crate) fn apply_rename_function(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameFunction,
    ) -> Result<(), EngineError> {
        if !cat.relational_functions.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "function \"{}\" does not exist",
                rename.old_name
            )));
        }
        if cat.relational_functions.contains_key(&rename.new_name) {
            return Err(EngineError::ApplyFailed(format!(
                "function \"{}\" already exists",
                rename.new_name
            )));
        }
        let Some(mut function) = cat.relational_functions.remove(&rename.old_name) else {
            return Ok(());
        };
        function.name = rename.new_name.clone();
        cat.relational_functions
            .insert(rename.new_name.clone(), function);
        let old_target = RelationalCommentTarget::Function {
            function: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::Function {
                    function: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }

    pub(crate) fn apply_create_sequence(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateSequence,
    ) -> Result<(), EngineError> {
        self.apply_create_sequence_with_replay_policy(cat, create, false)
    }

    pub(crate) fn apply_create_sequence_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateSequence,
    ) -> Result<(), EngineError> {
        self.apply_create_sequence_with_replay_policy(cat, create, true)
    }

    fn apply_create_sequence_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateSequence,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let target_exists = if legacy_replay {
            cat.relational_catalog.contains_key(&create.name)
                || cat.relational_views.contains_key(&create.name)
                || cat.relational_materialized_views.contains_key(&create.name)
                || cat.relational_sequences.contains_key(&create.name)
        } else {
            cat.pg_class_relation_kind(&create.name)?.is_some()
        };
        if target_exists {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.name
            )));
        }
        let mut next = cat.clone();
        let oid = if legacy_replay {
            let oid = next.relational_next_oid;
            next.relational_next_oid =
                next.relational_next_oid.checked_add(1).ok_or_else(|| {
                    EngineError::ApplyFailed(
                        "relational sequence OID allocation exhausted".to_string(),
                    )
                })?;
            oid
        } else {
            next.finalize_legacy_index_oid_migration()?;
            next.allocate_relational_class_oid("relational sequence OID allocation exhausted")?
        };
        next.relational_sequences.insert(
            create.name.clone(),
            RelationalSequence {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                last_value: 1,
                is_called: false,
                acl: BTreeMap::new(),
            },
        );
        *cat = next;
        Ok(())
    }

    pub(crate) fn preflight_create_domain(&self, create: &CreateDomain) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(&create.name)
            || cat.relational_views.contains_key(&create.name)
            || cat.relational_materialized_views.contains_key(&create.name)
            || cat.relational_sequences.contains_key(&create.name)
            || cat.relational_domains.contains_key(&create.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "type \"{}\" already exists",
                create.name
            )));
        }
        Ok(())
    }

    pub(crate) fn apply_create_domain(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateDomain,
    ) -> Result<(), EngineError> {
        self.preflight_create_domain(&create)?;
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational domain OID allocation exhausted".to_string())
        })?;
        cat.relational_domains.insert(
            create.name.clone(),
            RelationalDomain {
                schema: PUBLIC_SCHEMA_NAME.to_string(),
                name: create.name,
                oid,
                base_type: create.base_type,
            },
        );
        Ok(())
    }

    pub(crate) fn preflight_drop_domain(&self, drop: &DropDomain) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.domains {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "domain \"{}\" specified more than once",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_domains.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "domain \"{}\" does not exist",
                    name
                )));
            }
            if cat.relational_catalog.values().any(|table| {
                table
                    .columns
                    .iter()
                    .any(|column| column.domain.as_deref() == Some(name.as_str()))
            }) {
                return Err(EngineError::ApplyFailed(format!(
                    "cannot drop domain \"{}\" because other objects depend on it",
                    name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn apply_drop_domain(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropDomain,
    ) -> Result<(), EngineError> {
        self.preflight_drop_domain(&drop)?;
        for name in &drop.domains {
            cat.relational_domains.remove(name);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Domain {
                    domain: name.clone(),
                });
        }
        Ok(())
    }

    pub(crate) fn resolve_column_domain_type(
        &self,
        column: &mut ColumnDef,
    ) -> Result<(u32, i16), EngineError> {
        let cat = self.catalog_snapshot();
        if let Some(domain_name) = column.domain.as_ref() {
            let domain = cat.relational_domains.get(domain_name).ok_or_else(|| {
                EngineError::ApplyFailed(format!("type \"{}\" does not exist", domain_name))
            })?;
            column.ty = domain.base_type;
            Ok((domain.oid, domain.base_type.type_size()))
        } else {
            Ok((column.ty.postgres_oid(), column.ty.type_size()))
        }
    }

    pub(crate) fn preflight_implicit_sequence_name(&self, name: &str) -> Result<(), EngineError> {
        if self.apply_uses_legacy_index_semantics() {
            return self.preflight_implicit_sequence_name_legacy_replay(name);
        }
        let cat = self.catalog_snapshot();
        if cat.pg_class_relation_kind(name)?.is_some() {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{name}\" already exists"
            )));
        }
        Ok(())
    }

    pub(crate) fn preflight_implicit_sequence_name_legacy_replay(
        &self,
        name: &str,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(name)
            || cat.relational_views.contains_key(name)
            || cat.relational_materialized_views.contains_key(name)
            || cat.relational_sequences.contains_key(name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{name}\" already exists"
            )));
        }
        Ok(())
    }

    pub(crate) fn preflight_column_default_target(
        &self,
        default: &ColumnDefault,
    ) -> Result<(), EngineError> {
        self.preflight_column_default_target_with_replay_policy(default, false)
    }

    pub(crate) fn preflight_column_default_target_legacy_replay(
        &self,
        default: &ColumnDefault,
    ) -> Result<(), EngineError> {
        self.preflight_column_default_target_with_replay_policy(default, true)
    }

    fn preflight_column_default_target_with_replay_policy(
        &self,
        default: &ColumnDefault,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        match default {
            ColumnDefault::Literal(_) => Ok(()),
            ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing: true,
            } if legacy_replay => self.preflight_implicit_sequence_name_legacy_replay(sequence),
            ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing: true,
            } => self.preflight_implicit_sequence_name(sequence),
            ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing: false,
            } if legacy_replay => self.preflight_sequence_target_legacy_replay(sequence),
            ColumnDefault::SequenceNextVal {
                sequence,
                create_if_missing: false,
            } => self.preflight_sequence_target(sequence),
        }
    }

    pub(crate) fn apply_sequence_nextval(
        &self,
        cat: &mut DdlCatalogState,
        nextval: SequenceNextVal,
    ) -> Result<i64, EngineError> {
        let legacy_replay = self.apply_uses_legacy_index_semantics();
        self.apply_sequence_nextval_with_replay_policy(cat, nextval, legacy_replay)
    }

    pub(crate) fn apply_sequence_nextval_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        nextval: SequenceNextVal,
    ) -> Result<i64, EngineError> {
        self.apply_sequence_nextval_with_replay_policy(cat, nextval, true)
    }

    fn apply_sequence_nextval_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        nextval: SequenceNextVal,
        legacy_replay: bool,
    ) -> Result<i64, EngineError> {
        if legacy_replay {
            self.preflight_sequence_target_legacy_replay(&nextval.name)?;
        } else {
            self.preflight_sequence_target(&nextval.name)?;
        }
        let sequence = cat
            .relational_sequences
            .get_mut(&nextval.name)
            .expect("sequence target preflighted");
        let value = if sequence.is_called {
            sequence
                .last_value
                .checked_add(1)
                .ok_or_else(|| EngineError::ApplyFailed("sequence value overflow".to_string()))?
        } else {
            sequence.last_value
        };
        sequence.last_value = value;
        sequence.is_called = true;
        Ok(value)
    }

    pub(crate) fn evaluate_column_default(
        &self,
        cat: &mut DdlCatalogState,
        default: &ColumnDefault,
    ) -> Result<SqlValue, EngineError> {
        match default {
            ColumnDefault::Literal(value) => Ok(value.clone()),
            ColumnDefault::SequenceNextVal { sequence, .. } => {
                let value = self.apply_sequence_nextval(
                    cat,
                    SequenceNextVal {
                        name: sequence.clone(),
                    },
                )?;
                i32::try_from(value).map(SqlValue::Int4).map_err(|_| {
                    EngineError::ApplyFailed(
                        "sequence value is out of range for int4 default".to_string(),
                    )
                })
            }
        }
    }

    /// PURE column-default evaluation for `prepare_insert` (write-half MVCC, Stage 2). Identical
    /// arithmetic to [`Engine::evaluate_column_default`] / [`Engine::apply_sequence_nextval`], but
    /// `nextval` advances a per-call `seq_state` scratch (seeded lazily from the engine's sequence
    /// catalog) instead of mutating `self`. The scratch's final `(last_value, is_called)` per
    /// sequence is installed by `apply_delta`, so a prepare→apply pair advances the sequence by
    /// exactly what the old in-line apply did — while prepare stays `&self`.
    pub(crate) fn evaluate_column_default_pure(
        &self,
        default: &ColumnDefault,
        seq_state: &mut BTreeMap<String, (i64, bool)>,
    ) -> Result<SqlValue, EngineError> {
        match default {
            ColumnDefault::Literal(value) => Ok(value.clone()),
            ColumnDefault::SequenceNextVal { sequence, .. } => {
                self.preflight_sequence_target(sequence)?;
                let entry = seq_state.entry(sequence.clone()).or_insert_with(|| {
                    if let Some(state) =
                        self.current_transaction_read_snapshot()
                            .and_then(|snapshot| {
                                snapshot
                                    .delta
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                                    .sequence_state
                                    .get(sequence)
                                    .copied()
                            })
                    {
                        return state;
                    }
                    let catalog = self.catalog_snapshot();
                    let seq = catalog
                        .relational_sequences
                        .get(sequence)
                        .expect("sequence target preflighted");
                    (seq.last_value, seq.is_called)
                });
                let (last_value, is_called) = *entry;
                let value = if is_called {
                    last_value.checked_add(1).ok_or_else(|| {
                        EngineError::ApplyFailed("sequence value overflow".to_string())
                    })?
                } else {
                    last_value
                };
                *entry = (value, true);
                i32::try_from(value).map(SqlValue::Int4).map_err(|_| {
                    EngineError::ApplyFailed(
                        "sequence value is out of range for int4 default".to_string(),
                    )
                })
            }
        }
    }

    pub(crate) fn apply_sequence_setval(
        &self,
        cat: &mut DdlCatalogState,
        setval: SequenceSetVal,
    ) -> Result<i64, EngineError> {
        self.apply_sequence_setval_with_replay_policy(cat, setval, false)
    }

    pub(crate) fn apply_sequence_setval_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        setval: SequenceSetVal,
    ) -> Result<i64, EngineError> {
        self.apply_sequence_setval_with_replay_policy(cat, setval, true)
    }

    fn apply_sequence_setval_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        setval: SequenceSetVal,
        legacy_replay: bool,
    ) -> Result<i64, EngineError> {
        if legacy_replay {
            self.preflight_sequence_target_legacy_replay(&setval.name)?;
        } else {
            self.preflight_sequence_target(&setval.name)?;
        }
        let sequence = cat
            .relational_sequences
            .get_mut(&setval.name)
            .expect("sequence target preflighted");
        sequence.last_value = setval.value;
        sequence.is_called = setval.is_called;
        Ok(setval.value)
    }

    pub(crate) fn apply_create_schema(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateSchema,
    ) -> Result<(), EngineError> {
        if create.name != PUBLIC_SCHEMA_NAME {
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" is not supported",
                create.name
            )));
        }
        if cat.relational_public_schema_exists
            && !create.if_not_exists
            && !cat.relational_public_schema_implicit
        {
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" already exists",
                create.name
            )));
        }
        cat.relational_public_schema_exists = true;
        cat.relational_public_schema_implicit = false;
        Ok(())
    }

    pub(crate) fn apply_drop_schema(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSchema,
    ) -> Result<(), EngineError> {
        if drop.name != PUBLIC_SCHEMA_NAME {
            if drop.if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" does not exist",
                drop.name
            )));
        }
        if !cat.relational_public_schema_exists {
            if drop.if_exists {
                return Ok(());
            }
            return Err(EngineError::ApplyFailed(format!(
                "schema \"{}\" does not exist",
                drop.name
            )));
        }
        if !cat.relational_catalog.is_empty()
            || !cat.relational_views.is_empty()
            || !cat.relational_materialized_views.is_empty()
            || !cat.relational_functions.is_empty()
            || !cat.relational_sequences.is_empty()
            || !cat.relational_domains.is_empty()
            || !cat.relational_publications.is_empty()
            || !cat.relational_subscriptions.is_empty()
        {
            return Err(EngineError::ApplyFailed(format!(
                "cannot drop non-empty schema \"{}\"",
                drop.name
            )));
        }
        cat.relational_public_schema_exists = false;
        cat.relational_public_schema_implicit = false;
        cat.relational_schema_acl.clear();
        cat.relational_comments
            .remove(&RelationalCommentTarget::Schema { schema: drop.name });
        Ok(())
    }

    pub(crate) fn database_exists(&self, database: &str) -> bool {
        let cat = self.catalog_snapshot();
        database == "postgres" || cat.relational_databases.contains_key(database)
    }

    pub(crate) fn tablespace_exists(&self, tablespace: &str) -> bool {
        let cat = self.catalog_snapshot();
        matches!(tablespace, "pg_default" | "pg_global")
            || cat.relational_tablespaces.contains_key(tablespace)
    }

    pub(crate) fn apply_create_database(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateDatabase,
    ) -> Result<(), EngineError> {
        if self.database_exists(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "database \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational database OID allocation exhausted".to_string())
        })?;
        cat.relational_databases.insert(
            create.name.clone(),
            RelationalDatabase {
                name: create.name,
                oid,
                acl: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub(crate) fn apply_drop_database(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropDatabase,
    ) -> Result<(), EngineError> {
        let mut seen = BTreeSet::new();
        for database in &drop.names {
            if !seen.insert(database.clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "database \"{}\" specified more than once",
                    database
                )));
            }
            if database == "postgres" {
                return Err(EngineError::ApplyFailed(
                    "cannot drop bootstrap database \"postgres\"".to_string(),
                ));
            }
            if !drop.if_exists && !cat.relational_databases.contains_key(database) {
                return Err(EngineError::ApplyFailed(format!(
                    "database \"{}\" does not exist",
                    database
                )));
            }
        }
        for database in &drop.names {
            cat.relational_databases.remove(database);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Database {
                    database: database.clone(),
                });
        }
        Ok(())
    }

    pub(crate) fn apply_rename_database(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameDatabase,
    ) -> Result<(), EngineError> {
        if rename.old_name == "postgres" {
            return Err(EngineError::ApplyFailed(
                "cannot rename bootstrap database \"postgres\"".to_string(),
            ));
        }
        if !cat.relational_databases.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "database \"{}\" does not exist",
                rename.old_name
            )));
        }
        if self.database_exists(&rename.new_name) {
            return Err(EngineError::ApplyFailed(format!(
                "database \"{}\" already exists",
                rename.new_name
            )));
        }
        let mut database = cat
            .relational_databases
            .remove(&rename.old_name)
            .expect("database existence checked");
        database.name = rename.new_name.clone();
        cat.relational_databases
            .insert(rename.new_name.clone(), database);
        let old_target = RelationalCommentTarget::Database {
            database: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            let new_target = RelationalCommentTarget::Database {
                database: rename.new_name,
            };
            cat.relational_comments.insert(new_target, comment);
        }
        Ok(())
    }

    pub(crate) fn apply_create_tablespace(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateTablespace,
    ) -> Result<(), EngineError> {
        if self.tablespace_exists(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "tablespace \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational tablespace OID allocation exhausted".to_string())
        })?;
        cat.relational_tablespaces.insert(
            create.name.clone(),
            RelationalTablespace {
                name: create.name,
                oid,
                location: create.location,
                acl: BTreeMap::new(),
            },
        );
        Ok(())
    }

    pub(crate) fn apply_drop_tablespace(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropTablespace,
    ) -> Result<(), EngineError> {
        let mut seen = BTreeSet::new();
        for tablespace in &drop.names {
            if !seen.insert(tablespace.clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "tablespace \"{}\" specified more than once",
                    tablespace
                )));
            }
            if matches!(tablespace.as_str(), "pg_default" | "pg_global") {
                return Err(EngineError::ApplyFailed(format!(
                    "cannot drop bootstrap tablespace \"{}\"",
                    tablespace
                )));
            }
            if !drop.if_exists && !cat.relational_tablespaces.contains_key(tablespace) {
                return Err(EngineError::ApplyFailed(format!(
                    "tablespace \"{}\" does not exist",
                    tablespace
                )));
            }
        }
        for tablespace in &drop.names {
            cat.relational_tablespaces.remove(tablespace);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Tablespace {
                    tablespace: tablespace.clone(),
                });
        }
        Ok(())
    }

    pub(crate) fn apply_rename_tablespace(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameTablespace,
    ) -> Result<(), EngineError> {
        if matches!(rename.old_name.as_str(), "pg_default" | "pg_global") {
            return Err(EngineError::ApplyFailed(format!(
                "cannot rename bootstrap tablespace \"{}\"",
                rename.old_name
            )));
        }
        if !cat.relational_tablespaces.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "tablespace \"{}\" does not exist",
                rename.old_name
            )));
        }
        if self.tablespace_exists(&rename.new_name) {
            return Err(EngineError::ApplyFailed(format!(
                "tablespace \"{}\" already exists",
                rename.new_name
            )));
        }
        let mut tablespace = cat
            .relational_tablespaces
            .remove(&rename.old_name)
            .expect("tablespace existence checked");
        tablespace.name = rename.new_name.clone();
        cat.relational_tablespaces
            .insert(rename.new_name.clone(), tablespace);
        let old_target = RelationalCommentTarget::Tablespace {
            tablespace: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            let new_target = RelationalCommentTarget::Tablespace {
                tablespace: rename.new_name,
            };
            cat.relational_comments.insert(new_target, comment);
        }
        Ok(())
    }

    pub(crate) fn apply_drop_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropMaterializedView,
    ) -> Result<(), EngineError> {
        self.apply_drop_materialized_view_with_replay_policy(cat, drop, false)
    }

    pub(crate) fn apply_drop_materialized_view_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropMaterializedView,
    ) -> Result<(), EngineError> {
        self.apply_drop_materialized_view_with_replay_policy(cat, drop, true)
    }

    fn apply_drop_materialized_view_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropMaterializedView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        if legacy_replay {
            self.preflight_drop_materialized_view_legacy_replay(&drop)?;
        } else {
            self.preflight_drop_materialized_view(&drop)?;
        }
        for name in &drop.names {
            if cat.relational_materialized_views.remove(name).is_none() {
                continue;
            }
            cat.relational_comments
                .remove(&RelationalCommentTarget::MaterializedView {
                    materialized_view: name.clone(),
                });
        }
        Ok(())
    }

    pub(crate) fn apply_drop_sequence(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSequence,
    ) -> Result<(), EngineError> {
        self.apply_drop_sequence_with_replay_policy(cat, drop, false)
    }

    pub(crate) fn apply_drop_sequence_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSequence,
    ) -> Result<(), EngineError> {
        self.apply_drop_sequence_with_replay_policy(cat, drop, true)
    }

    fn apply_drop_sequence_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSequence,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        if legacy_replay {
            self.preflight_drop_sequence_legacy_replay(&drop)?;
        } else {
            self.preflight_drop_sequence(&drop)?;
        }
        for name in &drop.names {
            if cat.relational_sequences.remove(name).is_none() {
                continue;
            }
            cat.relational_comments
                .remove(&RelationalCommentTarget::Sequence {
                    sequence: name.clone(),
                });
        }
        Ok(())
    }
}
