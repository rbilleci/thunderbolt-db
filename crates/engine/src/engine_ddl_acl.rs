//! GRANT/REVOKE (ACL) + DDL preflight validation (P0 §9.6 decomposition,
//! behavior-preserving): a focused `impl Engine` block for the privilege
//! grant/revoke appliers across every object kind (table/relation/function/schema/
//! database/tablespace + default-table privileges) with their preflight ACL
//! target/grantee checks, plus the sequence/view/materialized-view preflights and
//! apply_rename_view.

use super::*;

impl Engine {
    pub(crate) fn preflight_create_sequence(
        &self,
        create: &CreateSequence,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.pg_class_relation_kind(&create.name)?.is_some() {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                create.name
            )));
        }
        Ok(())
    }

    pub(crate) fn preflight_sequence_target(&self, name: &str) -> Result<(), EngineError> {
        if self.apply_uses_legacy_index_semantics() {
            return self.preflight_sequence_target_legacy_replay(name);
        }
        let cat = self.catalog_snapshot();
        match cat.pg_class_relation_kind(name)? {
            Some(PgClassRelationKind::Sequence) => Ok(()),
            Some(_) => Err(EngineError::ApplyFailed(format!(
                "relation \"{name}\" is not a sequence"
            ))),
            None => Err(EngineError::ApplyFailed(format!(
                "sequence \"{name}\" does not exist"
            ))),
        }
    }

    pub(crate) fn preflight_sequence_target_legacy_replay(
        &self,
        name: &str,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(name)
            || cat.relational_views.contains_key(name)
            || cat.relational_materialized_views.contains_key(name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{name}\" is not a sequence"
            )));
        }
        if !cat.relational_sequences.contains_key(name) {
            return Err(EngineError::ApplyFailed(format!(
                "sequence \"{name}\" does not exist"
            )));
        }
        Ok(())
    }

    pub(crate) fn preflight_table_acl_target(&self, table: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_views.contains_key(table)
            || cat.relational_materialized_views.contains_key(table)
            || cat.relational_sequences.contains_key(table)
        {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{table}\" is not a table"
            )));
        }
        if !cat.relational_catalog.contains_key(table) {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{table}\" does not exist"
            )));
        }
        Ok(())
    }

    pub(crate) fn preflight_acl_target(
        &self,
        relation: &str,
        kind: AclRelationKind,
    ) -> Result<(), EngineError> {
        let actual = self.acl_relation_kind(relation).ok_or_else(|| {
            EngineError::ApplyFailed(format!("relation \"{relation}\" does not exist"))
        })?;
        let table_keyword_matches_relation = kind == AclRelationKind::Table
            && matches!(
                actual,
                AclRelationKind::Table | AclRelationKind::View | AclRelationKind::MaterializedView
            );
        if kind != AclRelationKind::Relation && kind != actual && !table_keyword_matches_relation {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{relation}\" is not a {}",
                acl_relation_kind_label(kind)
            )));
        }
        Ok(())
    }

    pub(crate) fn preflight_acl_grantee(&self, grantee: &str) -> Result<(), EngineError> {
        if self.role_exists(grantee) || grantee == "public" {
            Ok(())
        } else {
            Err(EngineError::ApplyFailed(format!(
                "role \"{grantee}\" does not exist"
            )))
        }
    }

    fn acl_relation_kind(&self, relation: &str) -> Option<AclRelationKind> {
        let cat = self.catalog_snapshot();
        if cat.relational_catalog.contains_key(relation) {
            Some(AclRelationKind::Table)
        } else if cat.relational_views.contains_key(relation) {
            Some(AclRelationKind::View)
        } else if cat.relational_materialized_views.contains_key(relation) {
            Some(AclRelationKind::MaterializedView)
        } else if cat.relational_sequences.contains_key(relation) {
            Some(AclRelationKind::Sequence)
        } else {
            None
        }
    }

    fn relational_acl_mut<'a>(
        cat: &'a mut DdlCatalogState,
        relation: &str,
    ) -> Option<&'a mut BTreeMap<String, BTreeSet<TablePrivilege>>> {
        if let Some(table) = cat.relational_catalog.get_mut(relation) {
            Some(&mut table.acl)
        } else if let Some(view) = cat.relational_views.get_mut(relation) {
            Some(&mut view.acl)
        } else if let Some(view) = cat.relational_materialized_views.get_mut(relation) {
            Some(&mut view.acl)
        } else if let Some(sequence) = cat.relational_sequences.get_mut(relation) {
            Some(&mut sequence.acl)
        } else {
            None
        }
    }

    pub(crate) fn apply_grant_acl(
        &self,
        cat: &mut DdlCatalogState,
        relation: &str,
        kind: AclRelationKind,
        grantee: &str,
        privileges: &[TablePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_acl_target(relation, kind)?;
        self.preflight_acl_grantee(grantee)?;
        let acl = Self::relational_acl_mut(cat, relation)
            .expect("relation ACL target preflighted")
            .entry(grantee.to_string())
            .or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    pub(crate) fn apply_revoke_acl(
        &self,
        cat: &mut DdlCatalogState,
        relation: &str,
        kind: AclRelationKind,
        grantee: &str,
        privileges: &[TablePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_acl_target(relation, kind)?;
        self.preflight_acl_grantee(grantee)?;
        let relation_acl =
            Self::relational_acl_mut(cat, relation).expect("relation ACL target preflighted");
        if let Some(acl) = relation_acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                relation_acl.remove(grantee);
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_function_acl_target(&self, function: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_functions.contains_key(function) {
            Ok(())
        } else {
            Err(EngineError::ApplyFailed(format!(
                "function \"{function}\" does not exist"
            )))
        }
    }

    pub(crate) fn apply_grant_function_acl(
        &self,
        cat: &mut DdlCatalogState,
        function: &str,
        grantee: &str,
        privileges: &[FunctionPrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_function_acl_target(function)?;
        self.preflight_acl_grantee(grantee)?;
        let acl = cat
            .relational_functions
            .get_mut(function)
            .expect("function ACL target preflighted")
            .acl
            .entry(grantee.to_string())
            .or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    pub(crate) fn apply_revoke_function_acl(
        &self,
        cat: &mut DdlCatalogState,
        function: &str,
        grantee: &str,
        privileges: &[FunctionPrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_function_acl_target(function)?;
        self.preflight_acl_grantee(grantee)?;
        let function = cat
            .relational_functions
            .get_mut(function)
            .expect("function ACL target preflighted");
        if let Some(acl) = function.acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                function.acl.remove(grantee);
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_schema_acl_target(&self, schema: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if schema != PUBLIC_SCHEMA_NAME || !cat.relational_public_schema_exists {
            return Err(EngineError::ApplyFailed(
                "schema does not exist".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn apply_grant_schema_acl(
        &self,
        cat: &mut DdlCatalogState,
        schema: &str,
        grantee: &str,
        privileges: &[SchemaPrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_schema_acl_target(schema)?;
        self.preflight_acl_grantee(grantee)?;
        let acl = cat
            .relational_schema_acl
            .entry(grantee.to_string())
            .or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    pub(crate) fn apply_revoke_schema_acl(
        &self,
        cat: &mut DdlCatalogState,
        schema: &str,
        grantee: &str,
        privileges: &[SchemaPrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_schema_acl_target(schema)?;
        self.preflight_acl_grantee(grantee)?;
        if let Some(acl) = cat.relational_schema_acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                cat.relational_schema_acl.remove(grantee);
            }
        }
        Ok(())
    }

    pub(crate) fn apply_grant_default_table_privileges(
        &self,
        cat: &mut DdlCatalogState,
        grantee: &str,
        privileges: &[TablePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_acl_grantee(grantee)?;
        let acl = cat
            .relational_default_table_acl
            .entry(grantee.to_string())
            .or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    pub(crate) fn apply_revoke_default_table_privileges(
        &self,
        cat: &mut DdlCatalogState,
        grantee: &str,
        privileges: &[TablePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_acl_grantee(grantee)?;
        if let Some(acl) = cat.relational_default_table_acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                cat.relational_default_table_acl.remove(grantee);
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_database_acl_target(&self, database: &str) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_databases.contains_key(database) {
            Ok(())
        } else {
            Err(EngineError::ApplyFailed(format!(
                "database \"{database}\" does not exist"
            )))
        }
    }

    pub(crate) fn apply_grant_database_acl(
        &self,
        cat: &mut DdlCatalogState,
        database: &str,
        grantee: &str,
        privileges: &[DatabasePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_database_acl_target(database)?;
        self.preflight_acl_grantee(grantee)?;
        let Some(database) = cat.relational_databases.get_mut(database) else {
            return Ok(());
        };
        let acl = database.acl.entry(grantee.to_string()).or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    pub(crate) fn apply_revoke_database_acl(
        &self,
        cat: &mut DdlCatalogState,
        database: &str,
        grantee: &str,
        privileges: &[DatabasePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_database_acl_target(database)?;
        self.preflight_acl_grantee(grantee)?;
        let Some(database) = cat.relational_databases.get_mut(database) else {
            return Ok(());
        };
        if let Some(acl) = database.acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                database.acl.remove(grantee);
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_tablespace_acl_target(
        &self,
        tablespace: &str,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_tablespaces.contains_key(tablespace) {
            Ok(())
        } else {
            Err(EngineError::ApplyFailed(format!(
                "tablespace \"{tablespace}\" does not exist"
            )))
        }
    }

    pub(crate) fn apply_grant_tablespace_acl(
        &self,
        cat: &mut DdlCatalogState,
        tablespace: &str,
        grantee: &str,
        privileges: &[TablespacePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_tablespace_acl_target(tablespace)?;
        self.preflight_acl_grantee(grantee)?;
        let Some(tablespace) = cat.relational_tablespaces.get_mut(tablespace) else {
            return Ok(());
        };
        let acl = tablespace.acl.entry(grantee.to_string()).or_default();
        for privilege in privileges {
            acl.insert(*privilege);
        }
        Ok(())
    }

    pub(crate) fn apply_revoke_tablespace_acl(
        &self,
        cat: &mut DdlCatalogState,
        tablespace: &str,
        grantee: &str,
        privileges: &[TablespacePrivilege],
    ) -> Result<(), EngineError> {
        self.preflight_tablespace_acl_target(tablespace)?;
        self.preflight_acl_grantee(grantee)?;
        let Some(tablespace) = cat.relational_tablespaces.get_mut(tablespace) else {
            return Ok(());
        };
        if let Some(acl) = tablespace.acl.get_mut(grantee) {
            for privilege in privileges {
                acl.remove(privilege);
            }
            if acl.is_empty() {
                tablespace.acl.remove(grantee);
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_create_materialized_view(
        &self,
        create: &CreateMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_create_materialized_view_with_replay_policy(create, false)
    }

    pub(crate) fn preflight_create_materialized_view_legacy_replay(
        &self,
        create: &CreateMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_create_materialized_view_with_replay_policy(create, true)
    }

    fn preflight_create_materialized_view_with_replay_policy(
        &self,
        create: &CreateMaterializedView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
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
        if legacy_replay {
            if cat.relational_views.contains_key(&create.query.table)
                || cat
                    .relational_materialized_views
                    .contains_key(&create.query.table)
            {
                return Err(EngineError::ApplyFailed(
                    "materialized views over views are unsupported".to_string(),
                ));
            }
            if !cat.relational_catalog.contains_key(&create.query.table) {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" does not exist",
                    create.query.table
                )));
            }
        } else {
            match cat.pg_class_relation_kind(&create.query.table)? {
                Some(PgClassRelationKind::Table) => {}
                Some(PgClassRelationKind::View | PgClassRelationKind::MaterializedView) => {
                    return Err(EngineError::ApplyFailed(
                        "materialized views over views are unsupported".to_string(),
                    ))
                }
                Some(_) => {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" cannot be a materialized-view dependency",
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
        Ok(())
    }

    pub(crate) fn preflight_refresh_materialized_view(
        &self,
        refresh: &RefreshMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_refresh_materialized_view_with_replay_policy(refresh, false)
    }

    pub(crate) fn preflight_refresh_materialized_view_legacy_replay(
        &self,
        refresh: &RefreshMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_refresh_materialized_view_with_replay_policy(refresh, true)
    }

    fn preflight_refresh_materialized_view_with_replay_policy(
        &self,
        refresh: &RefreshMaterializedView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let wrong_kind = if legacy_replay {
            cat.relational_catalog.contains_key(&refresh.name)
                || cat.relational_views.contains_key(&refresh.name)
                || cat.relational_sequences.contains_key(&refresh.name)
        } else {
            cat.pg_class_relation_kind(&refresh.name)?
                .is_some_and(|kind| kind != PgClassRelationKind::MaterializedView)
        };
        if wrong_kind {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a materialized view",
                refresh.name
            )));
        }
        if !cat
            .relational_materialized_views
            .contains_key(&refresh.name)
        {
            return Err(EngineError::ApplyFailed(format!(
                "materialized view \"{}\" does not exist",
                refresh.name
            )));
        }
        Ok(())
    }

    pub(crate) fn preflight_drop_view(&self, drop: &DropView) -> Result<(), EngineError> {
        self.preflight_drop_view_with_replay_policy(drop, false)
    }

    pub(crate) fn preflight_drop_view_legacy_replay(
        &self,
        drop: &DropView,
    ) -> Result<(), EngineError> {
        self.preflight_drop_view_with_replay_policy(drop, true)
    }

    fn preflight_drop_view_with_replay_policy(
        &self,
        drop: &DropView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        let drop_names = drop.names.iter().cloned().collect::<BTreeSet<_>>();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "view \"{}\" specified more than once",
                    name
                )));
            }
            let wrong_kind = if legacy_replay {
                cat.relational_catalog.contains_key(name)
                    || cat.relational_materialized_views.contains_key(name)
                    || cat.relational_sequences.contains_key(name)
            } else {
                cat.pg_class_relation_kind(name)?
                    .is_some_and(|kind| kind != PgClassRelationKind::View)
            };
            if wrong_kind {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a view",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_views.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "view \"{}\" does not exist",
                    name
                )));
            }
            if cat.relational_views.iter().any(|(candidate, _)| {
                !drop_names.contains(candidate) && self.relational_view_depends_on(candidate, name)
            }) {
                return Err(EngineError::ApplyFailed(format!(
                    "cannot drop view \"{}\" because another view depends on it",
                    name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_drop_materialized_view(
        &self,
        drop: &DropMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_drop_materialized_view_with_replay_policy(drop, false)
    }

    pub(crate) fn preflight_drop_materialized_view_legacy_replay(
        &self,
        drop: &DropMaterializedView,
    ) -> Result<(), EngineError> {
        self.preflight_drop_materialized_view_with_replay_policy(drop, true)
    }

    fn preflight_drop_materialized_view_with_replay_policy(
        &self,
        drop: &DropMaterializedView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "materialized view \"{}\" specified more than once",
                    name
                )));
            }
            let wrong_kind = if legacy_replay {
                cat.relational_catalog.contains_key(name)
                    || cat.relational_views.contains_key(name)
                    || cat.relational_sequences.contains_key(name)
            } else {
                cat.pg_class_relation_kind(name)?
                    .is_some_and(|kind| kind != PgClassRelationKind::MaterializedView)
            };
            if wrong_kind {
                return Err(EngineError::ApplyFailed(format!(
                    "relation \"{}\" is not a materialized view",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_materialized_views.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "materialized view \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn preflight_drop_sequence(&self, drop: &DropSequence) -> Result<(), EngineError> {
        self.preflight_drop_sequence_with_replay_policy(drop, false)
    }

    pub(crate) fn preflight_drop_sequence_legacy_replay(
        &self,
        drop: &DropSequence,
    ) -> Result<(), EngineError> {
        self.preflight_drop_sequence_with_replay_policy(drop, true)
    }

    fn preflight_drop_sequence_with_replay_policy(
        &self,
        drop: &DropSequence,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "sequence \"{}\" specified more than once",
                    name
                )));
            }
            if legacy_replay {
                if cat.relational_catalog.contains_key(name)
                    || cat.relational_views.contains_key(name)
                    || cat.relational_materialized_views.contains_key(name)
                {
                    return Err(EngineError::ApplyFailed(format!(
                        "relation \"{}\" is not a sequence",
                        name
                    )));
                }
                if !drop.if_exists && !cat.relational_sequences.contains_key(name) {
                    return Err(EngineError::ApplyFailed(format!(
                        "sequence \"{}\" does not exist",
                        name
                    )));
                }
            } else {
                match cat.pg_class_relation_kind(name)? {
                    Some(PgClassRelationKind::Sequence) => {}
                    Some(_) => {
                        return Err(EngineError::ApplyFailed(format!(
                            "relation \"{}\" is not a sequence",
                            name
                        )))
                    }
                    None if !drop.if_exists => {
                        return Err(EngineError::ApplyFailed(format!(
                            "sequence \"{}\" does not exist",
                            name
                        )))
                    }
                    None => {}
                }
            }
        }
        Ok(())
    }

    pub(crate) fn apply_rename_view(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameView,
    ) -> Result<(), EngineError> {
        self.apply_rename_view_with_replay_policy(cat, rename, false)
    }

    pub(crate) fn apply_rename_view_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameView,
    ) -> Result<(), EngineError> {
        self.apply_rename_view_with_replay_policy(cat, rename, true)
    }

    fn apply_rename_view_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let source_wrong_kind = if legacy_replay {
            cat.relational_catalog.contains_key(&rename.old_name)
                || cat
                    .relational_materialized_views
                    .contains_key(&rename.old_name)
                || cat.relational_sequences.contains_key(&rename.old_name)
        } else {
            cat.pg_class_relation_kind(&rename.old_name)?
                .is_some_and(|kind| kind != PgClassRelationKind::View)
        };
        if source_wrong_kind {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a view",
                rename.old_name
            )));
        }
        let destination_exists = if legacy_replay {
            cat.relational_catalog.contains_key(&rename.new_name)
                || cat.relational_views.contains_key(&rename.new_name)
                || cat
                    .relational_materialized_views
                    .contains_key(&rename.new_name)
                || cat.relational_sequences.contains_key(&rename.new_name)
        } else {
            cat.pg_class_relation_kind(&rename.new_name)?.is_some()
        };
        if destination_exists {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" already exists",
                rename.new_name
            )));
        }
        if self.relational_view_has_dependents(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "cannot rename view \"{}\" because another view depends on it",
                rename.old_name
            )));
        }
        let Some(mut view) = cat.relational_views.remove(&rename.old_name) else {
            return Err(EngineError::ApplyFailed(format!(
                "view \"{}\" does not exist",
                rename.old_name
            )));
        };
        view.name = rename.new_name.clone();
        cat.relational_views.insert(rename.new_name.clone(), view);

        let old_target = RelationalCommentTarget::View {
            view: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::View {
                    view: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }
}
