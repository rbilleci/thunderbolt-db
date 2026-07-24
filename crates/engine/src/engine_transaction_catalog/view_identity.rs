//! Typed identity and dependency closure for transaction-owned stored-view lifecycle operations.

use super::*;
use crate::engine_transaction_reset::table_schema_digest;

impl Engine {
    pub(crate) fn transaction_catalog_command_is_supported(command: &Command) -> bool {
        matches!(
            command,
            Command::CreateTable(_)
                | Command::CreateView(_)
                | Command::RenameView(_)
                | Command::DropView(_)
                | Command::CreateIndex(_)
                | Command::RenameIndex(_)
                | Command::DropIndex(_)
                | Command::CreateSequence(_)
                | Command::SequenceRestart(_)
                | Command::RenameSequence(_)
                | Command::DropSequence(_)
        )
    }

    pub(crate) fn apply_transaction_catalog_command(
        &self,
        working: &mut DdlCatalogState,
        command: Command,
    ) -> Result<(), EngineError> {
        // Every newly staged catalog command applies current semantics. The first admitted command
        // after a legacy prefix finalizes the one-way transition from recovery-only synthesized
        // index OIDs to the shared pg_class allocator and therefore selects op16/17. Once current,
        // index-neutral transactions retain their byte-stable op10/12/14 framing.
        working.finalize_legacy_index_oid_migration()?;
        match command {
            Command::CreateTable(create) => self.apply_create_table(working, create),
            Command::CreateView(create) => self.apply_create_view(working, create),
            Command::RenameView(rename) => self.apply_rename_view(working, rename),
            Command::DropView(drop) => self.apply_drop_view(working, drop),
            Command::CreateIndex(create) => self.apply_create_index_transactional(working, create),
            Command::RenameIndex(rename) => self.apply_rename_index(working, rename),
            Command::DropIndex(drop) => self.apply_drop_index(working, drop),
            Command::CreateSequence(create) => self.apply_create_sequence(working, create),
            Command::SequenceRestart(restart) => self.apply_restart_sequence(working, restart),
            Command::RenameSequence(rename) => self.apply_rename_sequence(working, rename),
            Command::DropSequence(drop) => self.apply_drop_sequence(working, drop),
            _ => Err(EngineError::ApplyFailed(
                "transaction catalog command is outside the admitted family".to_string(),
            )),
        }
    }

    fn apply_transaction_catalog_command_legacy_replay(
        &self,
        working: &mut DdlCatalogState,
        command: Command,
        record_seed: Index,
        command_ordinal: u32,
    ) -> Result<(), EngineError> {
        match command {
            Command::CreateTable(create) => {
                self.apply_create_table_legacy_replay(working, create, record_seed, command_ordinal)
            }
            Command::CreateView(create) => self.apply_create_view_legacy_replay(working, create),
            Command::RenameView(rename) => self.apply_rename_view_legacy_replay(working, rename),
            Command::DropView(drop) => self.apply_drop_view_legacy_replay(working, drop),
            Command::CreateIndex(create) => self.apply_create_index_legacy_replay(
                working,
                create,
                record_seed,
                command_ordinal,
                false,
            ),
            Command::RenameIndex(rename) => self.apply_rename_index_legacy_replay(working, rename),
            Command::DropIndex(drop) => self.apply_drop_index_legacy_replay(working, drop),
            _ => Err(EngineError::ApplyFailed(
                "legacy transaction catalog command is outside the admitted family".to_string(),
            )),
        }
    }

    pub(crate) fn transaction_view_operation_identity(
        command_index: u32,
        ordinal: u32,
        before: &CatalogSnapshot,
        after: &CatalogSnapshot,
        command: &Command,
    ) -> Result<BinaryTransactionViewLifecycleOperationIdentity, ExecuteError> {
        let targets = match command {
            Command::CreateView(create) => {
                let target_before = optional_view_preimage(before, &create.name)?;
                let dependencies = view_dependency_closure(before, &create.query.table)?;
                let target_after = required_view_postimage(after, &create.name)?;
                vec![BinaryTransactionViewLifecycleTargetIdentity {
                    before_name: create.name.clone(),
                    target_before,
                    dependencies,
                    after_name: Some(create.name.clone()),
                    target_after: Some(target_after),
                }]
            }
            Command::RenameView(rename) => {
                let view = required_view(before, &rename.old_name)?;
                ensure_relation_absent(before, &rename.new_name, "RENAME VIEW destination")?;
                let dependencies = view_dependency_closure(before, &view.query.table)?;
                vec![BinaryTransactionViewLifecycleTargetIdentity {
                    before_name: rename.old_name.clone(),
                    target_before: Some(view_relation_identity(view)?),
                    dependencies,
                    after_name: Some(rename.new_name.clone()),
                    target_after: Some(required_view_postimage(after, &rename.new_name)?),
                }]
            }
            Command::DropView(drop) => {
                let mut targets = Vec::with_capacity(drop.names.len());
                for name in &drop.names {
                    let (target_before, dependencies) =
                        if let Some(view) = before.relational_views.get(name) {
                            (
                                Some(view_relation_identity(view)?),
                                view_dependency_closure(before, &view.query.table)?,
                            )
                        } else {
                            ensure_relation_absent(before, name, "DROP VIEW target")?;
                            (None, BTreeMap::new())
                        };
                    ensure_relation_absent(after, name, "DROP VIEW postimage")?;
                    targets.push(BinaryTransactionViewLifecycleTargetIdentity {
                        before_name: name.clone(),
                        target_before,
                        dependencies,
                        after_name: None,
                        target_after: None,
                    });
                }
                targets
            }
            _ => {
                return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                    "transactional stored-view identity names an unsupported command".to_string(),
                )))
            }
        };
        let identity = BinaryTransactionViewLifecycleOperationIdentity {
            command_index,
            ordinal,
            targets,
        };
        if !valid_view_lifecycle_operation_identity(command, &identity) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "transactional stored-view identity is not canonical".to_string(),
            )));
        }
        Ok(identity)
    }

    pub(crate) fn validate_transaction_view_before(
        catalog: &CatalogSnapshot,
        command: &Command,
        identity: &BinaryTransactionViewLifecycleOperationIdentity,
    ) -> Result<(), EngineError> {
        if !valid_view_lifecycle_operation_identity(command, identity) {
            return Err(EngineError::Durability(
                "ordered stored-view operation carries a noncanonical identity".to_string(),
            ));
        }
        match command {
            Command::CreateView(create) => {
                let target = &identity.targets[0];
                let observed_before =
                    optional_view_preimage(catalog, &create.name).map_err(execute_to_engine)?;
                let observed_dependencies = view_dependency_closure(catalog, &create.query.table)
                    .map_err(execute_to_engine)?;
                validate_view_preimage(target, observed_before, observed_dependencies)
            }
            Command::RenameView(rename) => {
                let target = &identity.targets[0];
                let view = required_view(catalog, &rename.old_name).map_err(execute_to_engine)?;
                ensure_relation_absent(catalog, &rename.new_name, "RENAME VIEW destination")
                    .map_err(execute_to_engine)?;
                let dependencies = view_dependency_closure(catalog, &view.query.table)
                    .map_err(execute_to_engine)?;
                validate_view_preimage(
                    target,
                    Some(view_relation_identity(view).map_err(execute_to_engine)?),
                    dependencies,
                )
            }
            Command::DropView(drop) => {
                for (name, target) in drop.names.iter().zip(&identity.targets) {
                    let (observed_before, dependencies) = if let Some(view) =
                        optional_view_binding(catalog, name).map_err(execute_to_engine)?
                    {
                        (
                            Some(view_relation_identity(view).map_err(execute_to_engine)?),
                            view_dependency_closure(catalog, &view.query.table)
                                .map_err(execute_to_engine)?,
                        )
                    } else {
                        ensure_relation_absent(catalog, name, "DROP VIEW target")
                            .map_err(execute_to_engine)?;
                        (None, BTreeMap::new())
                    };
                    validate_view_preimage(target, observed_before, dependencies)?;
                }
                Ok(())
            }
            _ => Err(EngineError::Durability(
                "ordered stored-view identity names an unsupported command".to_string(),
            )),
        }
    }

    pub(crate) fn validate_transaction_view_after(
        catalog: &CatalogSnapshot,
        command: &Command,
        identity: &BinaryTransactionViewLifecycleOperationIdentity,
    ) -> Result<(), EngineError> {
        match command {
            Command::CreateView(create) => validate_view_postimage(
                &identity.targets[0],
                &create.name,
                required_view_postimage(catalog, &create.name).map_err(execute_to_engine)?,
            ),
            Command::RenameView(rename) => {
                ensure_relation_absent(catalog, &rename.old_name, "RENAME VIEW old binding")
                    .map_err(execute_to_engine)?;
                validate_view_postimage(
                    &identity.targets[0],
                    &rename.new_name,
                    required_view_postimage(catalog, &rename.new_name)
                        .map_err(execute_to_engine)?,
                )
            }
            Command::DropView(drop) => {
                for name in &drop.names {
                    ensure_relation_absent(catalog, name, "DROP VIEW postimage")
                        .map_err(execute_to_engine)?;
                }
                if identity
                    .targets
                    .iter()
                    .any(|target| target.after_name.is_some() || target.target_after.is_some())
                {
                    return Err(EngineError::Durability(
                        "ordered DROP VIEW carries a non-absent postimage".to_string(),
                    ));
                }
                Ok(())
            }
            _ => Err(EngineError::Durability(
                "ordered stored-view identity names an unsupported command".to_string(),
            )),
        }
    }

    fn validate_transaction_view_before_legacy(
        catalog: &CatalogSnapshot,
        command: &Command,
        identity: &BinaryTransactionViewLifecycleOperationIdentity,
    ) -> Result<(), EngineError> {
        if !valid_view_lifecycle_operation_identity(command, identity) {
            return Err(EngineError::Durability(
                "legacy ordered stored-view operation carries a noncanonical identity".to_string(),
            ));
        }
        match command {
            Command::CreateView(create) => {
                let target = &identity.targets[0];
                let observed_before = optional_view_preimage_legacy(catalog, &create.name)
                    .map_err(execute_to_engine)?;
                let observed_dependencies =
                    view_dependency_closure_legacy(catalog, &create.query.table)
                        .map_err(execute_to_engine)?;
                validate_view_preimage(target, observed_before, observed_dependencies)
            }
            Command::RenameView(rename) => {
                let target = &identity.targets[0];
                let view =
                    required_view_legacy(catalog, &rename.old_name).map_err(execute_to_engine)?;
                ensure_relation_absent_legacy(catalog, &rename.new_name, "RENAME VIEW destination")
                    .map_err(execute_to_engine)?;
                let dependencies = view_dependency_closure_legacy(catalog, &view.query.table)
                    .map_err(execute_to_engine)?;
                validate_view_preimage(
                    target,
                    Some(view_relation_identity(view).map_err(execute_to_engine)?),
                    dependencies,
                )
            }
            Command::DropView(drop) => {
                for (name, target) in drop.names.iter().zip(&identity.targets) {
                    let (observed_before, dependencies) =
                        if let Some(view) = catalog.relational_views.get(name) {
                            (
                                Some(view_relation_identity(view).map_err(execute_to_engine)?),
                                view_dependency_closure_legacy(catalog, &view.query.table)
                                    .map_err(execute_to_engine)?,
                            )
                        } else {
                            ensure_relation_absent_legacy(catalog, name, "DROP VIEW target")
                                .map_err(execute_to_engine)?;
                            (None, BTreeMap::new())
                        };
                    validate_view_preimage(target, observed_before, dependencies)?;
                }
                Ok(())
            }
            _ => Err(EngineError::Durability(
                "legacy ordered stored-view identity names an unsupported command".to_string(),
            )),
        }
    }

    fn validate_transaction_view_after_legacy(
        catalog: &CatalogSnapshot,
        command: &Command,
        identity: &BinaryTransactionViewLifecycleOperationIdentity,
    ) -> Result<(), EngineError> {
        match command {
            Command::CreateView(create) => validate_view_postimage(
                &identity.targets[0],
                &create.name,
                required_view_postimage_legacy(catalog, &create.name).map_err(execute_to_engine)?,
            ),
            Command::RenameView(rename) => {
                ensure_relation_absent_legacy(catalog, &rename.old_name, "RENAME VIEW old binding")
                    .map_err(execute_to_engine)?;
                validate_view_postimage(
                    &identity.targets[0],
                    &rename.new_name,
                    required_view_postimage_legacy(catalog, &rename.new_name)
                        .map_err(execute_to_engine)?,
                )
            }
            Command::DropView(drop) => {
                for name in &drop.names {
                    ensure_relation_absent_legacy(catalog, name, "DROP VIEW postimage")
                        .map_err(execute_to_engine)?;
                }
                if identity
                    .targets
                    .iter()
                    .any(|target| target.after_name.is_some() || target.target_after.is_some())
                {
                    return Err(EngineError::Durability(
                        "legacy ordered DROP VIEW carries a non-absent postimage".to_string(),
                    ));
                }
                Ok(())
            }
            _ => Err(EngineError::Durability(
                "legacy ordered stored-view identity names an unsupported command".to_string(),
            )),
        }
    }

    pub(crate) fn apply_transaction_catalog_envelope(
        &self,
        working: &mut DdlCatalogState,
        commit_seq: Index,
        catalog_epoch: BinaryTransactionCatalogEpoch,
        record_seed: Index,
        commands: &[BinaryTransactionCatalogCommand],
        envelope: TransactionCatalogEnvelopeSlices<'_>,
    ) -> Result<(), EngineError> {
        let TransactionCatalogEnvelopeSlices {
            view_operations,
            view_lifecycle_operations,
            index_lifecycle_operations,
            sequence_lifecycle_operations,
            sequence_reset_operations,
            operation_order,
            catalog_output,
            sequence_input_oids,
        } = envelope;
        let mut prior_reset_ordinal = None;
        for identity in sequence_reset_operations {
            if !valid_sequence_reset_operation_identity(identity)
                || prior_reset_ordinal.is_some_and(|prior| identity.ordinal <= prior)
                || usize::try_from(identity.ordinal)
                    .ok()
                    .and_then(|ordinal| operation_order.get(ordinal))
                    .is_none_or(|operation| {
                        !matches!(
                            operation,
                            BinaryTransactionOperationIdentity::TableReset { table }
                                if table == &identity.table
                        )
                    })
            {
                return Err(EngineError::Durability(
                    "ordered transaction carries a misplaced sequence reset identity".to_string(),
                ));
            }
            prior_reset_ordinal = Some(identity.ordinal);
        }
        if !view_operations.is_empty() && !view_lifecycle_operations.is_empty() {
            return Err(EngineError::Durability(
                "ordered transaction mixes legacy and lifecycle view identities".to_string(),
            ));
        }
        let legacy_lifecycle;
        let view_operations = if view_lifecycle_operations.is_empty() {
            legacy_lifecycle = view_operations
                .iter()
                .map(|identity| {
                    let operation = commands
                        .get(usize::try_from(identity.command_index).map_err(|_| {
                            EngineError::Durability(
                                "legacy view command index exceeds usize".to_string(),
                            )
                        })?)
                        .ok_or_else(|| {
                            EngineError::Durability(
                                "legacy view identity has no catalog command".to_string(),
                            )
                        })?;
                    let Command::CreateView(create) = &operation.command else {
                        return Err(EngineError::Durability(
                            "legacy view identity names a non-CREATE command".to_string(),
                        ));
                    };
                    Ok(lifecycle_from_legacy_create(create, identity))
                })
                .collect::<Result<Vec<_>, EngineError>>()?;
            legacy_lifecycle.as_slice()
        } else {
            view_lifecycle_operations
        };
        // Historical opcodes remain byte-stable after the one-way index-identity boundary. Their
        // namespace/allocation semantics therefore come from the already-published working epoch:
        // an opcode 12/14 before that boundary is genuinely legacy, while the same byte layout
        // emitted after the boundary must use current shared-pg_class rules. Opcodes 16/17 carry
        // the boundary explicitly.
        let current_catalog_semantics = catalog_epoch
            == BinaryTransactionCatalogEpoch::IndexIdentityV1
            || working.index_oid_epoch_current;
        let mut next_view = 0usize;
        let mut next_index = 0usize;
        let mut next_sequence = 0usize;
        let mut next_sequence_reset = 0usize;
        for (command_index, operation) in commands.iter().enumerate() {
            while sequence_reset_operations
                .get(next_sequence_reset)
                .is_some_and(|identity| identity.ordinal < operation.ordinal)
            {
                Self::apply_transaction_sequence_reset_identity(
                    working,
                    commit_seq,
                    &sequence_reset_operations[next_sequence_reset],
                )?;
                next_sequence_reset += 1;
            }
            if sequence_reset_operations
                .get(next_sequence_reset)
                .is_some_and(|identity| identity.ordinal == operation.ordinal)
            {
                return Err(EngineError::Durability(
                    "catalog command and sequence reset reuse one statement ordinal".to_string(),
                ));
            }
            let before = Self::catalog_snapshot_from_working(working, commit_seq);
            match &operation.command {
                Command::CreateTable(_) => {
                    if view_operations.get(next_view).is_some_and(|identity| {
                        usize::try_from(identity.command_index).ok() == Some(command_index)
                    }) {
                        return Err(EngineError::Durability(
                            "ordered CREATE TABLE carries a stored-view identity".to_string(),
                        ));
                    }
                    if index_lifecycle_operations
                        .get(next_index)
                        .is_some_and(|identity| {
                            usize::try_from(identity.command_index).ok() == Some(command_index)
                        })
                    {
                        return Err(EngineError::Durability(
                            "ordered CREATE TABLE carries an index identity".to_string(),
                        ));
                    }
                    if sequence_lifecycle_operations
                        .get(next_sequence)
                        .is_some_and(|identity| {
                            usize::try_from(identity.command_index).ok() == Some(command_index)
                        })
                    {
                        return Err(EngineError::Durability(
                            "ordered CREATE TABLE carries a sequence identity".to_string(),
                        ));
                    }
                }
                command if command_is_view_lifecycle(command) => {
                    let identity = view_operations.get(next_view).ok_or_else(|| {
                        EngineError::Durability(
                            "ordered stored-view operation lost its typed identity closure"
                                .to_string(),
                        )
                    })?;
                    if usize::try_from(identity.command_index).ok() != Some(command_index)
                        || identity.ordinal != operation.ordinal
                    {
                        return Err(EngineError::Durability(
                            "ordered stored-view identity changed command position".to_string(),
                        ));
                    }
                    if current_catalog_semantics {
                        Self::validate_transaction_view_before(&before, command, identity)?;
                    } else {
                        Self::validate_transaction_view_before_legacy(&before, command, identity)?;
                    }
                }
                command
                    if command_is_index_lifecycle(command)
                        && catalog_epoch == BinaryTransactionCatalogEpoch::IndexIdentityV1 =>
                {
                    let identity = index_lifecycle_operations.get(next_index).ok_or_else(|| {
                        EngineError::Durability(
                            "ordered index operation lost its typed identity closure".to_string(),
                        )
                    })?;
                    if usize::try_from(identity.command_index).ok() != Some(command_index)
                        || identity.ordinal != operation.ordinal
                    {
                        return Err(EngineError::Durability(
                            "ordered index identity changed command position".to_string(),
                        ));
                    }
                    Self::validate_transaction_index_before(&before, command, identity)?;
                }
                command
                    if command_is_sequence_lifecycle(command)
                        && catalog_epoch == BinaryTransactionCatalogEpoch::IndexIdentityV1 =>
                {
                    let identity = sequence_lifecycle_operations
                        .get(next_sequence)
                        .ok_or_else(|| {
                            EngineError::Durability(
                                "ordered sequence operation lost its typed identity closure"
                                    .to_string(),
                            )
                        })?;
                    if usize::try_from(identity.command_index).ok() != Some(command_index)
                        || identity.ordinal != operation.ordinal
                    {
                        return Err(EngineError::Durability(
                            "ordered sequence identity changed command position".to_string(),
                        ));
                    }
                    Self::validate_transaction_sequence_before(&before, command, identity)?;
                }
                _ => {
                    return Err(EngineError::Durability(
                        "transaction WAL record contains an unsupported catalog operation"
                            .to_string(),
                    ))
                }
            }
            self.with_apply_catalog(Some(Arc::clone(&before)), || {
                if current_catalog_semantics {
                    self.apply_transaction_catalog_command(working, operation.command.clone())
                } else {
                    self.apply_transaction_catalog_command_legacy_replay(
                        working,
                        operation.command.clone(),
                        record_seed,
                        operation.ordinal,
                    )
                }
            })?;
            if let (Some(output), Command::CreateTable(create)) =
                (catalog_output, &operation.command)
            {
                for (sequence, generated) in
                    create
                        .columns
                        .iter()
                        .filter_map(|column| match &column.default {
                            Some(ColumnDefault::SequenceNextVal {
                                sequence,
                                create_if_missing,
                            }) => Some((sequence, *create_if_missing)),
                            _ => None,
                        })
                {
                    let expected_oid = sequence_input_oids
                        .get(&(operation.ordinal, sequence.clone()))
                        .ok_or_else(|| {
                            EngineError::Durability(format!(
                                "ordered CREATE TABLE sequence input \"{sequence}\" lost its creator-local identity"
                            ))
                        })?;
                    if working
                        .relational_sequences
                        .get(sequence)
                        .map(|observed| observed.oid)
                        != Some(*expected_oid)
                    {
                        return Err(EngineError::Durability(format!(
                            "ordered CREATE TABLE sequence input \"{sequence}\" changed creator-local stable identity"
                        )));
                    }
                    if generated && output.created_sequence_oids.get(sequence) != Some(expected_oid)
                    {
                        return Err(EngineError::Durability(format!(
                            "ordered CREATE TABLE generated sequence \"{sequence}\" changed creator-local output identity"
                        )));
                    }
                }
            }
            if command_is_view_lifecycle(&operation.command) {
                let identity = &view_operations[next_view];
                let after = Self::catalog_snapshot_from_working(working, commit_seq);
                if current_catalog_semantics {
                    Self::validate_transaction_view_after(&after, &operation.command, identity)?;
                } else {
                    Self::validate_transaction_view_after_legacy(
                        &after,
                        &operation.command,
                        identity,
                    )?;
                }
                next_view += 1;
            }
            if catalog_epoch == BinaryTransactionCatalogEpoch::IndexIdentityV1
                && command_is_index_lifecycle(&operation.command)
            {
                let identity = &index_lifecycle_operations[next_index];
                let after = Self::catalog_snapshot_from_working(working, commit_seq);
                Self::validate_transaction_index_after(&after, &operation.command, identity)?;
                next_index += 1;
            }
            if catalog_epoch == BinaryTransactionCatalogEpoch::IndexIdentityV1
                && command_is_sequence_lifecycle(&operation.command)
            {
                let identity = &sequence_lifecycle_operations[next_sequence];
                let after = Self::catalog_snapshot_from_working(working, commit_seq);
                Self::validate_transaction_sequence_after(&after, &operation.command, identity)?;
                next_sequence += 1;
            }
        }
        while let Some(identity) = sequence_reset_operations.get(next_sequence_reset) {
            Self::apply_transaction_sequence_reset_identity(working, commit_seq, identity)?;
            next_sequence_reset += 1;
        }
        if next_view != view_operations.len() {
            return Err(EngineError::Durability(
                "ordered transaction carries an unbound stored-view identity".to_string(),
            ));
        }
        if next_index != index_lifecycle_operations.len() {
            return Err(EngineError::Durability(
                "ordered transaction carries an unbound index identity".to_string(),
            ));
        }
        if next_sequence != sequence_lifecycle_operations.len() {
            return Err(EngineError::Durability(
                "ordered transaction carries an unbound sequence identity".to_string(),
            ));
        }
        if next_sequence_reset != sequence_reset_operations.len() {
            return Err(EngineError::Durability(
                "ordered transaction carries an unbound sequence reset identity".to_string(),
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_transaction_catalog_before_wal(
        &self,
        record_catalog_epoch: BinaryTransactionCatalogEpoch,
        commands: &[BinaryTransactionCatalogCommand],
        envelope: TransactionCatalogEnvelopeSlices<'_>,
        expected: &CatalogSnapshot,
    ) -> Result<(), ExecuteError> {
        let TransactionCatalogEnvelopeSlices {
            view_operations: _,
            view_lifecycle_operations: _,
            index_lifecycle_operations: _,
            sequence_lifecycle_operations: _,
            sequence_reset_operations,
            operation_order: _,
            catalog_output: _,
            sequence_input_oids: _,
        } = envelope;
        let mut working = self.ddl_catalog().clone();
        let catalog_epoch = if commands.iter().any(|operation| {
            command_requires_index_catalog_opcode(&operation.command)
                || command_is_sequence_lifecycle(&operation.command)
        }) || !sequence_reset_operations.is_empty()
            || (!commands.is_empty() && !working.index_oid_epoch_current)
        {
            BinaryTransactionCatalogEpoch::IndexIdentityV1
        } else {
            BinaryTransactionCatalogEpoch::Legacy
        };
        if record_catalog_epoch != catalog_epoch {
            return Err(ExecuteError::Serialization(
                "typed transactional catalog epoch no longer matches its allocator transition"
                    .to_string(),
            ));
        }
        self.apply_transaction_catalog_envelope(
            &mut working,
            expected.commit_seq,
            catalog_epoch,
            expected.commit_seq,
            commands,
            envelope,
        )
        .map_err(ExecuteError::Engine)?;
        let reconstructed = Self::catalog_snapshot_from_working(&working, expected.commit_seq);
        if !reconstructed.same_contents(expected) {
            return Err(ExecuteError::Serialization(
                "typed transactional catalog envelope no longer reconstructs its exact private post-state"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_view_preimage(
    target: &BinaryTransactionViewLifecycleTargetIdentity,
    observed_before: Option<BinaryCatalogRelationIdentity>,
    observed_dependencies: BTreeMap<String, BinaryCatalogRelationIdentity>,
) -> Result<(), EngineError> {
    if observed_before != target.target_before {
        return Err(EngineError::Durability(format!(
            "ordered stored-view target {:?} changed stable preimage",
            target.before_name
        )));
    }
    if observed_dependencies != target.dependencies {
        return Err(EngineError::Durability(format!(
            "ordered stored-view target {:?} changed transitive source identity closure",
            target.before_name
        )));
    }
    Ok(())
}

fn validate_view_postimage(
    target: &BinaryTransactionViewLifecycleTargetIdentity,
    name: &str,
    observed: BinaryCatalogRelationIdentity,
) -> Result<(), EngineError> {
    if target.after_name.as_deref() != Some(name) || target.target_after.as_ref() != Some(&observed)
    {
        return Err(EngineError::Durability(format!(
            "ordered stored-view target {name:?} changed stable postimage"
        )));
    }
    Ok(())
}

fn optional_view_preimage(
    catalog: &CatalogSnapshot,
    name: &str,
) -> Result<Option<BinaryCatalogRelationIdentity>, ExecuteError> {
    optional_view_binding(catalog, name)?
        .map(view_relation_identity)
        .transpose()
}

fn optional_view_binding<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<Option<&'a RelationalView>, ExecuteError> {
    match catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
    {
        Some(PgClassRelationKind::View) => {
            catalog.relational_views.get(name).map(Some).ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(format!(
                    "catalog view binding {name:?} disappeared"
                )))
            })
        }
        Some(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "relation {name:?} is not a view"
        )))),
        None => Ok(None),
    }
}

fn required_view<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<&'a RelationalView, ExecuteError> {
    match catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
    {
        Some(PgClassRelationKind::View) => catalog.relational_views.get(name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(format!(
                "catalog view binding {name:?} disappeared"
            )))
        }),
        Some(_) => Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "relation {name:?} is not a view"
        )))),
        None => Err(ExecuteError::UndefinedRelation(format!(
            "transactional stored-view target {name:?}"
        ))),
    }
}

fn required_view_postimage(
    catalog: &CatalogSnapshot,
    name: &str,
) -> Result<BinaryCatalogRelationIdentity, ExecuteError> {
    required_view(catalog, name).and_then(view_relation_identity)
}

fn ensure_relation_absent(
    catalog: &CatalogSnapshot,
    name: &str,
    context: &str,
) -> Result<(), ExecuteError> {
    if catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
        .is_some()
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "transactional {context} {name:?} is not absent"
        ))));
    }
    Ok(())
}

fn optional_view_preimage_legacy(
    catalog: &CatalogSnapshot,
    name: &str,
) -> Result<Option<BinaryCatalogRelationIdentity>, ExecuteError> {
    if let Some(view) = catalog.relational_views.get(name) {
        return view_relation_identity(view).map(Some);
    }
    ensure_relation_absent_legacy(catalog, name, "stored-view target")?;
    Ok(None)
}

fn required_view_legacy<'a>(
    catalog: &'a CatalogSnapshot,
    name: &str,
) -> Result<&'a RelationalView, ExecuteError> {
    if catalog.relational_catalog.contains_key(name)
        || catalog.relational_materialized_views.contains_key(name)
        || catalog.relational_sequences.contains_key(name)
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "relation {name:?} is not a view"
        ))));
    }
    catalog
        .relational_views
        .get(name)
        .ok_or_else(|| ExecuteError::UndefinedRelation(format!("stored-view target {name:?}")))
}

fn required_view_postimage_legacy(
    catalog: &CatalogSnapshot,
    name: &str,
) -> Result<BinaryCatalogRelationIdentity, ExecuteError> {
    required_view_legacy(catalog, name).and_then(view_relation_identity)
}

fn ensure_relation_absent_legacy(
    catalog: &CatalogSnapshot,
    name: &str,
    context: &str,
) -> Result<(), ExecuteError> {
    if catalog.relational_catalog.contains_key(name)
        || catalog.relational_views.contains_key(name)
        || catalog.relational_materialized_views.contains_key(name)
        || catalog.relational_sequences.contains_key(name)
    {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
            "legacy transactional {context} {name:?} is not absent"
        ))));
    }
    Ok(())
}

pub(crate) fn view_semantic_digest(
    view: &RelationalView,
) -> Result<gpu_db_wal::CanonicalDigest, ExecuteError> {
    let query = serde_json::to_vec(&view.query).map_err(|error| {
        ExecuteError::Engine(EngineError::Durability(format!(
            "view query cannot be encoded for durable identity: {error}"
        )))
    })?;
    let mut body = Vec::new();
    body.extend_from_slice(b"GPUDBVIEWIDENTITY1");
    push_string(&mut body, &view.schema)?;
    push_string(&mut body, &view.name)?;
    body.extend_from_slice(&view.oid.to_le_bytes());
    push_bytes(&mut body, &query)?;
    push_string(&mut body, &view.definition)?;
    push_len(&mut body, view.acl.len())?;
    for (grantee, privileges) in &view.acl {
        push_string(&mut body, grantee)?;
        push_len(&mut body, privileges.len())?;
        for privilege in privileges {
            body.push(match privilege {
                TablePrivilege::Select => 1,
                TablePrivilege::Insert => 2,
                TablePrivilege::Update => 3,
                TablePrivilege::Delete => 4,
            });
        }
    }
    Ok(gpu_db_wal::canonical_request_digest(&body))
}

fn view_relation_identity(
    view: &RelationalView,
) -> Result<BinaryCatalogRelationIdentity, ExecuteError> {
    Ok(BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::View,
        oid: view.oid,
        digest: view_semantic_digest(view)?,
    })
}

fn table_relation_identity(
    table: &RelationalTable,
) -> Result<BinaryCatalogRelationIdentity, ExecuteError> {
    Ok(BinaryCatalogRelationIdentity {
        kind: BinaryCatalogRelationKind::Table,
        oid: table.oid,
        digest: table_schema_digest(table)?,
    })
}

fn view_dependency_closure(
    catalog: &CatalogSnapshot,
    source: &str,
) -> Result<BTreeMap<String, BinaryCatalogRelationIdentity>, ExecuteError> {
    let mut closure = BTreeMap::new();
    let mut visiting = BTreeSet::new();
    collect_view_dependency(catalog, source, &mut visiting, &mut closure)?;
    Ok(closure)
}

fn view_dependency_closure_legacy(
    catalog: &CatalogSnapshot,
    source: &str,
) -> Result<BTreeMap<String, BinaryCatalogRelationIdentity>, ExecuteError> {
    let mut closure = BTreeMap::new();
    let mut visiting = BTreeSet::new();
    collect_view_dependency_legacy(catalog, source, &mut visiting, &mut closure)?;
    Ok(closure)
}

fn collect_view_dependency(
    catalog: &CatalogSnapshot,
    name: &str,
    visiting: &mut BTreeSet<String>,
    closure: &mut BTreeMap<String, BinaryCatalogRelationIdentity>,
) -> Result<(), ExecuteError> {
    let view = match catalog
        .pg_class_relation_kind(name)
        .map_err(ExecuteError::Engine)?
    {
        Some(PgClassRelationKind::Table) => {
            let table = catalog.relational_catalog.get(name).ok_or_else(|| {
                ExecuteError::Engine(EngineError::Durability(format!(
                    "catalog table binding {name:?} disappeared"
                )))
            })?;
            closure.insert(name.to_string(), table_relation_identity(table)?);
            return Ok(());
        }
        Some(PgClassRelationKind::View) => catalog.relational_views.get(name).ok_or_else(|| {
            ExecuteError::Engine(EngineError::Durability(format!(
                "catalog view binding {name:?} disappeared"
            )))
        })?,
        Some(_) => {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(format!(
                "relation {name:?} cannot be a stored-view dependency"
            ))))
        }
        None => {
            return Err(ExecuteError::UndefinedRelation(format!(
                "transactional CREATE VIEW dependency {name:?}"
            )))
        }
    };
    if !visiting.insert(name.to_string()) {
        return Err(ExecuteError::Engine(EngineError::ApplyFailed(
            "view dependency cycle is unsupported".to_string(),
        )));
    }
    closure.insert(name.to_string(), view_relation_identity(view)?);
    collect_view_dependency(catalog, &view.query.table, visiting, closure)?;
    visiting.remove(name);
    Ok(())
}

fn collect_view_dependency_legacy(
    catalog: &CatalogSnapshot,
    name: &str,
    visiting: &mut BTreeSet<String>,
    closure: &mut BTreeMap<String, BinaryCatalogRelationIdentity>,
) -> Result<(), ExecuteError> {
    if let Some(view) = catalog.relational_views.get(name) {
        if !visiting.insert(name.to_string()) {
            return Err(ExecuteError::Engine(EngineError::ApplyFailed(
                "view dependency cycle is unsupported".to_string(),
            )));
        }
        closure.insert(name.to_string(), view_relation_identity(view)?);
        collect_view_dependency_legacy(catalog, &view.query.table, visiting, closure)?;
        visiting.remove(name);
        return Ok(());
    }
    if let Some(table) = catalog.relational_catalog.get(name) {
        closure.insert(name.to_string(), table_relation_identity(table)?);
        return Ok(());
    }
    Err(ExecuteError::UndefinedRelation(format!(
        "legacy transactional CREATE VIEW dependency {name:?}"
    )))
}

fn push_len(body: &mut Vec<u8>, len: usize) -> Result<(), ExecuteError> {
    let len = u64::try_from(len).map_err(|_| {
        ExecuteError::Unsupported("view identity count exceeds durable framing".to_string())
    })?;
    body.extend_from_slice(&len.to_le_bytes());
    Ok(())
}

fn push_bytes(body: &mut Vec<u8>, value: &[u8]) -> Result<(), ExecuteError> {
    push_len(body, value.len())?;
    body.extend_from_slice(value);
    Ok(())
}

fn push_string(body: &mut Vec<u8>, value: &str) -> Result<(), ExecuteError> {
    push_bytes(body, value.as_bytes())
}

fn execute_to_engine(error: ExecuteError) -> EngineError {
    match error {
        ExecuteError::Engine(error) => error,
        other => EngineError::Durability(other.to_string()),
    }
}
