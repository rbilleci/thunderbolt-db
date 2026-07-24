//! Publication / subscription / role DDL (P0 §9.6 decomposition, behavior-
//! preserving): a focused `impl Engine` block for CREATE/DROP PUBLICATION +
//! SUBSCRIPTION (with preflights), CREATE/DROP/RENAME ROLE (+ role_exists /
//! role_has_dependencies), and the sequence + materialized-view rename appliers.

use super::*;

impl Engine {
    pub(crate) fn preflight_create_publication(
        &self,
        create: &CreatePublication,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_publications.contains_key(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "publication \"{}\" already exists",
                create.name
            )));
        }
        if let PublicationTarget::Tables(tables) = &create.target {
            let mut seen = BTreeSet::new();
            for table in tables {
                if !seen.insert(table) {
                    return Err(EngineError::ApplyFailed(format!(
                        "table \"{}\" specified more than once",
                        table
                    )));
                }
                self.preflight_table_acl_target(table)?;
            }
        }
        Ok(())
    }

    pub(crate) fn apply_create_publication(
        &self,
        cat: &mut DdlCatalogState,
        create: CreatePublication,
    ) -> Result<(), EngineError> {
        self.preflight_create_publication(&create)?;
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational publication OID allocation exhausted".to_string())
        })?;
        let (all_tables, tables) = match create.target {
            PublicationTarget::AllTables => (true, Vec::new()),
            PublicationTarget::Tables(tables) => (false, tables),
        };
        cat.relational_publications.insert(
            create.name.clone(),
            RelationalPublication {
                name: create.name,
                oid,
                all_tables,
                tables,
            },
        );
        Ok(())
    }

    pub(crate) fn role_exists(&self, role: &str) -> bool {
        let cat = self.catalog_snapshot();
        role == "postgres" || cat.relational_roles.contains_key(role)
    }

    pub(crate) fn apply_create_role(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateRole,
    ) -> Result<(), EngineError> {
        if create.name == "postgres" || cat.relational_roles.contains_key(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "role \"{}\" already exists",
                create.name
            )));
        }
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational OID counter overflow".to_string())
        })?;
        cat.relational_roles.insert(
            create.name.clone(),
            RelationalRole {
                name: create.name,
                oid,
                login: create.login,
            },
        );
        Ok(())
    }

    pub(crate) fn apply_alter_role_login(
        &self,
        cat: &mut DdlCatalogState,
        alter: AlterRoleLogin,
    ) -> Result<(), EngineError> {
        if alter.name == "postgres" {
            return Err(EngineError::ApplyFailed(
                "cannot alter bootstrap role \"postgres\"".to_string(),
            ));
        }
        let role = cat.relational_roles.get_mut(&alter.name).ok_or_else(|| {
            EngineError::ApplyFailed(format!("role \"{}\" does not exist", alter.name))
        })?;
        role.login = alter.login;
        Ok(())
    }

    pub(crate) fn role_has_dependencies(&self, role: &str) -> bool {
        let cat = self.catalog_snapshot();
        cat.relational_comments
            .contains_key(&RelationalCommentTarget::Role {
                role: role.to_string(),
            })
            || cat
                .relational_catalog
                .values()
                .any(|table| table.acl.contains_key(role))
            || cat
                .relational_views
                .values()
                .any(|view| view.acl.contains_key(role))
            || cat
                .relational_materialized_views
                .values()
                .any(|view| view.acl.contains_key(role))
            || cat
                .relational_sequences
                .values()
                .any(|sequence| sequence.acl.contains_key(role))
            || cat
                .relational_databases
                .values()
                .any(|database| database.acl.contains_key(role))
            || cat
                .relational_tablespaces
                .values()
                .any(|tablespace| tablespace.acl.contains_key(role))
            || cat
                .relational_functions
                .values()
                .any(|function| function.acl.contains_key(role))
            || cat.relational_schema_acl.contains_key(role)
            || cat.relational_default_table_acl.contains_key(role)
    }

    pub(crate) fn apply_drop_role(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropRole,
    ) -> Result<(), EngineError> {
        let mut seen = BTreeSet::new();
        for role in &drop.names {
            if !seen.insert(role.clone()) {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" specified more than once",
                    role
                )));
            }
            if role == "postgres" {
                return Err(EngineError::ApplyFailed(
                    "cannot drop bootstrap role \"postgres\"".to_string(),
                ));
            }
            if !drop.if_exists && !cat.relational_roles.contains_key(role) {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" does not exist",
                    role
                )));
            }
            if cat.relational_roles.contains_key(role) && self.role_has_dependencies(role) {
                return Err(EngineError::ApplyFailed(format!(
                    "role \"{}\" cannot be dropped because dependent metadata exists",
                    role
                )));
            }
        }
        for role in drop.names {
            cat.relational_roles.remove(&role);
        }
        Ok(())
    }

    pub(crate) fn apply_rename_role(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameRole,
    ) -> Result<(), EngineError> {
        if rename.old_name == "postgres" {
            return Err(EngineError::ApplyFailed(
                "cannot rename bootstrap role \"postgres\"".to_string(),
            ));
        }
        if !cat.relational_roles.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "role \"{}\" does not exist",
                rename.old_name
            )));
        }
        if self.role_exists(&rename.new_name) {
            return Err(EngineError::ApplyFailed(format!(
                "role \"{}\" already exists",
                rename.new_name
            )));
        }
        let mut role = cat
            .relational_roles
            .remove(&rename.old_name)
            .expect("role existence checked");
        role.name = rename.new_name.clone();
        cat.relational_roles.insert(rename.new_name.clone(), role);
        let old_target = RelationalCommentTarget::Role {
            role: rename.old_name.clone(),
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            let new_target = RelationalCommentTarget::Role {
                role: rename.new_name.clone(),
            };
            cat.relational_comments.insert(new_target, comment);
        }
        for table in cat.relational_catalog.values_mut() {
            if let Some(privileges) = table.acl.remove(&rename.old_name) {
                table.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for view in cat.relational_views.values_mut() {
            if let Some(privileges) = view.acl.remove(&rename.old_name) {
                view.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for view in cat.relational_materialized_views.values_mut() {
            if let Some(privileges) = view.acl.remove(&rename.old_name) {
                view.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for sequence in cat.relational_sequences.values_mut() {
            if let Some(privileges) = sequence.acl.remove(&rename.old_name) {
                sequence.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for database in cat.relational_databases.values_mut() {
            if let Some(privileges) = database.acl.remove(&rename.old_name) {
                database.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for tablespace in cat.relational_tablespaces.values_mut() {
            if let Some(privileges) = tablespace.acl.remove(&rename.old_name) {
                tablespace.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        for function in cat.relational_functions.values_mut() {
            if let Some(privileges) = function.acl.remove(&rename.old_name) {
                function.acl.insert(rename.new_name.clone(), privileges);
            }
        }
        if let Some(privileges) = cat.relational_schema_acl.remove(&rename.old_name) {
            cat.relational_schema_acl
                .insert(rename.new_name.clone(), privileges);
        }
        if let Some(privileges) = cat.relational_default_table_acl.remove(&rename.old_name) {
            cat.relational_default_table_acl
                .insert(rename.new_name, privileges);
        }
        Ok(())
    }

    pub(crate) fn preflight_drop_publication(
        &self,
        drop: &DropPublication,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "publication \"{}\" specified more than once",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_publications.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "publication \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn apply_drop_publication(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropPublication,
    ) -> Result<(), EngineError> {
        self.preflight_drop_publication(&drop)?;
        for name in &drop.names {
            cat.relational_publications.remove(name);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Publication {
                    publication: name.clone(),
                });
        }
        Ok(())
    }

    pub(crate) fn preflight_create_subscription(
        &self,
        create: &CreateSubscription,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        if cat.relational_subscriptions.contains_key(&create.name) {
            return Err(EngineError::ApplyFailed(format!(
                "subscription \"{}\" already exists",
                create.name
            )));
        }
        let mut seen = BTreeSet::new();
        for publication in &create.publications {
            if !seen.insert(publication) {
                return Err(EngineError::ApplyFailed(format!(
                    "publication \"{}\" specified more than once",
                    publication
                )));
            }
            if !cat.relational_publications.contains_key(publication) {
                return Err(EngineError::ApplyFailed(format!(
                    "publication \"{}\" does not exist",
                    publication
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn apply_create_subscription(
        &self,
        cat: &mut DdlCatalogState,
        create: CreateSubscription,
    ) -> Result<(), EngineError> {
        self.preflight_create_subscription(&create)?;
        let oid = cat.relational_next_oid;
        cat.relational_next_oid = cat.relational_next_oid.checked_add(1).ok_or_else(|| {
            EngineError::ApplyFailed("relational subscription OID allocation exhausted".to_string())
        })?;
        cat.relational_subscriptions.insert(
            create.name.clone(),
            RelationalSubscription {
                name: create.name,
                oid,
                connection: create.connection,
                publications: create.publications,
                enabled: false,
            },
        );
        Ok(())
    }

    pub(crate) fn preflight_drop_subscription(
        &self,
        drop: &DropSubscription,
    ) -> Result<(), EngineError> {
        let cat = self.catalog_snapshot();
        let mut seen = BTreeSet::new();
        for name in &drop.names {
            if !seen.insert(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "subscription \"{}\" specified more than once",
                    name
                )));
            }
            if !drop.if_exists && !cat.relational_subscriptions.contains_key(name) {
                return Err(EngineError::ApplyFailed(format!(
                    "subscription \"{}\" does not exist",
                    name
                )));
            }
        }
        Ok(())
    }

    pub(crate) fn apply_drop_subscription(
        &self,
        cat: &mut DdlCatalogState,
        drop: DropSubscription,
    ) -> Result<(), EngineError> {
        self.preflight_drop_subscription(&drop)?;
        for name in &drop.names {
            cat.relational_subscriptions.remove(name);
            cat.relational_comments
                .remove(&RelationalCommentTarget::Subscription {
                    subscription: name.clone(),
                });
        }
        Ok(())
    }

    pub(crate) fn apply_rename_sequence(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameSequence,
    ) -> Result<(), EngineError> {
        self.apply_rename_sequence_with_replay_policy(cat, rename, false)
    }

    pub(crate) fn apply_rename_sequence_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameSequence,
    ) -> Result<(), EngineError> {
        self.apply_rename_sequence_with_replay_policy(cat, rename, true)
    }

    fn apply_rename_sequence_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameSequence,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let source_wrong_kind = if legacy_replay {
            cat.relational_catalog.contains_key(&rename.old_name)
                || cat.relational_views.contains_key(&rename.old_name)
                || cat
                    .relational_materialized_views
                    .contains_key(&rename.old_name)
        } else {
            cat.pg_class_relation_kind(&rename.old_name)?
                .is_some_and(|kind| kind != PgClassRelationKind::Sequence)
        };
        if source_wrong_kind {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a sequence",
                rename.old_name
            )));
        }
        if !cat.relational_sequences.contains_key(&rename.old_name) {
            return Err(EngineError::ApplyFailed(format!(
                "sequence \"{}\" does not exist",
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
        let Some(mut sequence) = cat.relational_sequences.remove(&rename.old_name) else {
            return Ok(());
        };
        sequence.name = rename.new_name.clone();
        cat.relational_sequences
            .insert(rename.new_name.clone(), sequence);

        let old_target = RelationalCommentTarget::Sequence {
            sequence: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::Sequence {
                    sequence: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }

    pub(crate) fn apply_rename_materialized_view(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameMaterializedView,
    ) -> Result<(), EngineError> {
        self.apply_rename_materialized_view_with_replay_policy(cat, rename, false)
    }

    pub(crate) fn apply_rename_materialized_view_legacy_replay(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameMaterializedView,
    ) -> Result<(), EngineError> {
        self.apply_rename_materialized_view_with_replay_policy(cat, rename, true)
    }

    fn apply_rename_materialized_view_with_replay_policy(
        &self,
        cat: &mut DdlCatalogState,
        rename: RenameMaterializedView,
        legacy_replay: bool,
    ) -> Result<(), EngineError> {
        let source_wrong_kind = if legacy_replay {
            cat.relational_catalog.contains_key(&rename.old_name)
                || cat.relational_views.contains_key(&rename.old_name)
                || cat.relational_sequences.contains_key(&rename.old_name)
        } else {
            cat.pg_class_relation_kind(&rename.old_name)?
                .is_some_and(|kind| kind != PgClassRelationKind::MaterializedView)
        };
        if source_wrong_kind {
            return Err(EngineError::ApplyFailed(format!(
                "relation \"{}\" is not a materialized view",
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
        let Some(mut view) = cat.relational_materialized_views.remove(&rename.old_name) else {
            return Err(EngineError::ApplyFailed(format!(
                "materialized view \"{}\" does not exist",
                rename.old_name
            )));
        };
        view.name = rename.new_name.clone();
        cat.relational_materialized_views
            .insert(rename.new_name.clone(), view);

        let old_target = RelationalCommentTarget::MaterializedView {
            materialized_view: rename.old_name,
        };
        if let Some(comment) = cat.relational_comments.remove(&old_target) {
            cat.relational_comments.insert(
                RelationalCommentTarget::MaterializedView {
                    materialized_view: rename.new_name,
                },
                comment,
            );
        }
        Ok(())
    }
}
