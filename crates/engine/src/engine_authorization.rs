//! Exact-generation authorization through a terminal device verdict.
//!
//! Session identity remains control-plane metadata, but no host catalog map decides whether a
//! request is allowed.  Each check serializes the complete immutable catalog generation as raw
//! role/object/grant candidates, uploads it with the typed request, and maps only the terminal GPU
//! verdict to PostgreSQL diagnostics.  This keeps rename stable by role OID and makes a missing
//! target fall through to the ordinary object-resolution diagnostic.

use super::*;
use gpu_db_execution::{
    DeviceAclGrant, DeviceAclObject, DeviceAclRequest, DeviceAclRole, DeviceAclSchemaGrant,
    DeviceAclVerdict, DeviceRoleNameRequest, DeviceRoleNameVerdict,
};

const ACL_TARGET_RELATION: u32 = 1;
const ACL_TARGET_SEQUENCE: u32 = 2;
const ACL_TARGET_FUNCTION: u32 = 3;
const ACL_TARGET_SCHEMA: u32 = 4;
const ACL_TARGET_SYSTEM: u32 = 5;
const ACL_TARGET_CONTROL: u32 = 6;

const ACL_PRIV_SELECT: u32 = 1;
const ACL_PRIV_INSERT: u32 = 2;
const ACL_PRIV_UPDATE: u32 = 3;
const ACL_PRIV_DELETE: u32 = 4;

const ACL_SCHEMA_USAGE: u32 = 1;
const ACL_SCHEMA_CREATE: u32 = 2;

const ACL_BOOTSTRAP_OID: u32 = PG_BOOTSTRAP_OWNER_OID as u32;
const ACL_PUBLIC_SCHEMA_OID: u32 = 2_200;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AuthorizationPrincipal {
    #[default]
    BootstrapPostgres,
    RoleOid(u32),
}

impl AuthorizationPrincipal {
    pub fn is_bootstrap(self) -> bool {
        matches!(self, Self::BootstrapPostgres)
    }

    fn oid(self) -> u32 {
        match self {
            Self::BootstrapPostgres => ACL_BOOTSTRAP_OID,
            Self::RoleOid(oid) => oid,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum AuthorizationTarget {
    Relation,
    Sequence,
    Function,
    Schema,
    Control,
}

impl AuthorizationTarget {
    fn code(self) -> u32 {
        match self {
            Self::Relation => ACL_TARGET_RELATION,
            Self::Sequence => ACL_TARGET_SEQUENCE,
            Self::Function => ACL_TARGET_FUNCTION,
            Self::Schema => ACL_TARGET_SCHEMA,
            Self::Control => ACL_TARGET_CONTROL,
        }
    }

    fn denial(self) -> ExecuteError {
        let object = match self {
            Self::Function => "function",
            Self::Schema => "schema",
            Self::Relation | Self::Sequence | Self::Control => "relation",
        };
        ExecuteError::PermissionDenied(format!("permission denied for {object}"))
    }
}

#[derive(Default)]
struct DeviceAclFacts {
    bytes: Vec<u8>,
    roles: Vec<DeviceAclRole>,
    objects: Vec<DeviceAclObject>,
    grants: Vec<DeviceAclGrant>,
    schema_grants: Vec<DeviceAclSchemaGrant>,
}

impl DeviceAclFacts {
    fn text(&mut self, value: &str) -> Result<(u32, u32), ExecuteError> {
        let offset = u32::try_from(self.bytes.len()).map_err(|_| acl_input_error())?;
        let len = u32::try_from(value.len()).map_err(|_| acl_input_error())?;
        self.bytes.extend_from_slice(value.as_bytes());
        Ok((offset, len))
    }

    fn object(&mut self, oid: u32, kind: u32, name: &str) -> Result<(), ExecuteError> {
        let (name_offset, name_len) = self.text(name)?;
        self.objects.push(DeviceAclObject {
            oid,
            kind,
            name_offset,
            name_len,
        });
        Ok(())
    }

    /// User objects retain both their search-path shorthand and explicit public-schema identity.
    /// The two names share one stable OID and one complete grant relation; an explicit
    /// `public.pg_class` can therefore never collide with the separate system candidate while a
    /// bare `pg_class` remains detectably ambiguous and fails closed.
    fn user_object(&mut self, oid: u32, kind: u32, name: &str) -> Result<(), ExecuteError> {
        self.object(oid, kind, name)?;
        if !name.starts_with("public.") {
            self.object(oid, kind, &format!("public.{name}"))?;
        }
        Ok(())
    }

    fn table_grants(
        &mut self,
        oid: u32,
        kind: u32,
        acl: &BTreeMap<String, BTreeSet<TablePrivilege>>,
    ) -> Result<(), ExecuteError> {
        for (grantee, privileges) in acl {
            let (grantee_name_offset, grantee_name_len) = self.text(grantee)?;
            let grantee_is_public = u32::from(grantee == "public");
            for privilege in privileges {
                self.grants.push(DeviceAclGrant {
                    object_oid: oid,
                    object_kind: kind,
                    grantee_name_offset,
                    grantee_name_len,
                    privilege: table_privilege_code(*privilege),
                    grantee_is_public,
                });
            }
        }
        Ok(())
    }

    fn function_grants(
        &mut self,
        oid: u32,
        acl: &BTreeMap<String, BTreeSet<FunctionPrivilege>>,
    ) -> Result<(), ExecuteError> {
        for (grantee, privileges) in acl {
            let (grantee_name_offset, grantee_name_len) = self.text(grantee)?;
            let grantee_is_public = u32::from(grantee == "public");
            for privilege in privileges {
                self.grants.push(DeviceAclGrant {
                    object_oid: oid,
                    object_kind: ACL_TARGET_FUNCTION,
                    grantee_name_offset,
                    grantee_name_len,
                    privilege: function_privilege_code(*privilege),
                    grantee_is_public,
                });
            }
        }
        Ok(())
    }
}

impl Engine {
    /// Resolve a session role at one published generation through the complete device role
    /// relation.  SET ROLE consumes this result directly, so this must not use a host catalog-map
    /// lookup that later authorization happens to revalidate.
    pub fn resolve_authorization_principal(
        &self,
        role: &str,
    ) -> Result<AuthorizationPrincipal, ExecuteError> {
        self.resolve_authorization_principal_at(None, role)
    }

    /// Session-aware form used by SQL-visible `SET [LOCAL] ROLE`.  An active transaction owns an
    /// immutable overlay generation, so role resolution must consume that exact generation rather
    /// than the concurrently published catalog snapshot.
    pub fn resolve_authorization_principal_at(
        &self,
        txn_id: Option<TxnId>,
        role: &str,
    ) -> Result<AuthorizationPrincipal, ExecuteError> {
        let catalog = match txn_id.and_then(|txn_id| self.transaction_snapshot_handle(txn_id)) {
            Some(snapshot) => snapshot.transaction_catalog(),
            None => self.catalog_snapshot(),
        };
        let mut facts = device_acl_facts(&catalog)?;
        let (name_offset, name_len) = facts.text(role)?;
        let anchor = catalog_relation_table(
            "pg_catalog",
            "__device_role_name_verdict_anchor",
            &[("anchor", SqlType::Int4)],
        );
        let (_, memory) =
            self.build_transient_relation_residency(&anchor, &[vec![SqlValue::Int4(0)]])?;
        let verdict = memory
            .catalog_role_name_verdict(
                DeviceRoleNameRequest {
                    name_offset,
                    name_len,
                },
                &facts.roles,
                &facts.bytes,
            )
            .map_err(|error| acl_device_failure(error.to_string()))?;
        #[cfg(test)]
        let mut sabotage = DEVICE_ROLE_NAME_SABOTAGE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        #[cfg(test)]
        if sabotage.as_ref().is_some_and(|request| {
            request.engine_identity == self as *const Self as usize && request.role == role
        }) {
            *sabotage = None;
            return Err(acl_device_failure(
                "injected terminal device role-name verdict failure",
            ));
        }
        match verdict {
            DeviceRoleNameVerdict::Found(ACL_BOOTSTRAP_OID) => {
                Ok(AuthorizationPrincipal::BootstrapPostgres)
            }
            DeviceRoleNameVerdict::Found(oid) => Ok(AuthorizationPrincipal::RoleOid(oid)),
            DeviceRoleNameVerdict::Missing => Err(ExecuteError::UndefinedRole(role.to_string())),
            DeviceRoleNameVerdict::InvalidInput => Err(acl_device_failure(
                "device role-name relation rejected malformed or duplicate candidates",
            )),
        }
    }

    pub fn authorize_relation_for_session(
        &self,
        txn_id: Option<TxnId>,
        principal: AuthorizationPrincipal,
        relation: &str,
        privilege: TablePrivilege,
    ) -> Result<(), ExecuteError> {
        let catalog = match txn_id.and_then(|txn_id| self.transaction_snapshot_handle(txn_id)) {
            Some(snapshot) => snapshot.transaction_catalog(),
            None => self.catalog_snapshot(),
        };
        self.authorize_device_target(
            &catalog,
            principal,
            relation,
            AuthorizationTarget::Relation,
            table_privilege_code(privilege),
        )
    }

    /// Authorize a sequence operation against the sequence candidate relation.  Sequence ACLs
    /// are intentionally distinct from table/view ACLs in the device request, even though both
    /// reuse the bounded table-privilege encoding.
    pub fn authorize_sequence_for_session(
        &self,
        txn_id: Option<TxnId>,
        principal: AuthorizationPrincipal,
        sequence: &str,
        privilege: TablePrivilege,
    ) -> Result<(), ExecuteError> {
        let catalog = match txn_id.and_then(|txn_id| self.transaction_snapshot_handle(txn_id)) {
            Some(snapshot) => snapshot.transaction_catalog(),
            None => self.catalog_snapshot(),
        };
        self.authorize_device_target(
            &catalog,
            principal,
            sequence,
            AuthorizationTarget::Sequence,
            table_privilege_code(privilege),
        )
    }

    pub(crate) fn authorize_command_at(
        &self,
        catalog: &CatalogSnapshot,
        principal: AuthorizationPrincipal,
        command: &Command,
    ) -> Result<(), ExecuteError> {
        authorize_command(self, catalog, principal, command)
    }

    pub(crate) fn authorize_select_at(
        &self,
        catalog: &CatalogSnapshot,
        principal: AuthorizationPrincipal,
        select: &Select,
    ) -> Result<(), ExecuteError> {
        // The parser retains explicit-public provenance separately from the residency key.  Put it
        // back into the typed device request so `public.pg_class` matches only the public object
        // candidate, while the bare `pg_class` request remains deliberately ambiguous.
        let target = if select.public_only {
            format!("public.{}", select.table)
        } else {
            select.table.clone()
        };
        self.authorize_device_target(
            catalog,
            principal,
            &target,
            AuthorizationTarget::Relation,
            ACL_PRIV_SELECT,
        )
    }

    pub(crate) fn authorize_function_at(
        &self,
        catalog: &CatalogSnapshot,
        principal: AuthorizationPrincipal,
        function: &str,
    ) -> Result<(), ExecuteError> {
        self.authorize_device_target(
            catalog,
            principal,
            function,
            AuthorizationTarget::Function,
            ACL_PRIV_SELECT,
        )
    }

    pub(crate) fn authorize_relation_names_at(
        &self,
        catalog: &CatalogSnapshot,
        principal: AuthorizationPrincipal,
        relations: &[String],
    ) -> Result<(), ExecuteError> {
        for relation in relations {
            self.authorize_device_target(
                catalog,
                principal,
                relation,
                AuthorizationTarget::Relation,
                ACL_PRIV_SELECT,
            )?;
        }
        Ok(())
    }

    fn authorize_device_target(
        &self,
        catalog: &CatalogSnapshot,
        principal: AuthorizationPrincipal,
        target: &str,
        target_kind: AuthorizationTarget,
        privilege: u32,
    ) -> Result<(), ExecuteError> {
        let mut facts = device_acl_facts(catalog)?;
        let (target_name_offset, target_name_len) = facts.text(target)?;
        let request = DeviceAclRequest {
            principal_oid: principal.oid(),
            bootstrap: u32::from(principal.is_bootstrap()),
            target_kind: target_kind.code(),
            privilege,
            target_name_offset,
            target_name_len,
        };
        #[cfg(test)]
        let mut sabotage = DEVICE_ACL_SABOTAGE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        #[cfg(test)]
        if sabotage.as_ref().is_some_and(|request| {
            request.engine_identity == self as *const Self as usize && request.target == target
        }) {
            *sabotage = None;
            return Err(acl_device_failure(
                "injected terminal device verdict failure",
            ));
        }
        // The anchor has no policy information.  It only owns the exact GPU context/stream while
        // the complete raw candidate buffers are uploaded and evaluated by the typed operator.
        let anchor = catalog_relation_table(
            "pg_catalog",
            "__device_acl_verdict_anchor",
            &[("anchor", SqlType::Int4)],
        );
        let (_, memory) =
            self.build_transient_relation_residency(&anchor, &[vec![SqlValue::Int4(0)]])?;
        let verdict = memory
            .catalog_acl_verdict(
                request,
                &facts.roles,
                &facts.objects,
                &facts.grants,
                &facts.schema_grants,
                &facts.bytes,
            )
            .map_err(|error| acl_device_failure(error.to_string()))?;
        match verdict {
            DeviceAclVerdict::Allowed | DeviceAclVerdict::MissingObject => Ok(()),
            DeviceAclVerdict::DeniedObject => Err(target_kind.denial()),
            DeviceAclVerdict::DeniedSchema => Err(AuthorizationTarget::Schema.denial()),
            DeviceAclVerdict::InvalidInput => Err(acl_device_failure(
                "terminal device verdict rejected malformed catalog candidates",
            )),
        }
    }

    #[cfg(test)]
    pub(crate) fn sabotage_next_device_acl_verdict(&self, target: &str) {
        *DEVICE_ACL_SABOTAGE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(DeviceAclSabotage {
            engine_identity: self as *const Self as usize,
            target: target.to_string(),
        });
    }

    #[cfg(test)]
    pub(crate) fn sabotage_next_device_role_name_verdict(&self, role: &str) {
        *DEVICE_ROLE_NAME_SABOTAGE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(DeviceRoleNameSabotage {
            engine_identity: self as *const Self as usize,
            role: role.to_string(),
        });
    }
}

#[cfg(test)]
struct DeviceAclSabotage {
    engine_identity: usize,
    target: String,
}

#[cfg(test)]
struct DeviceRoleNameSabotage {
    engine_identity: usize,
    role: String,
}

#[cfg(test)]
static DEVICE_ACL_SABOTAGE: std::sync::Mutex<Option<DeviceAclSabotage>> =
    std::sync::Mutex::new(None);
#[cfg(test)]
static DEVICE_ROLE_NAME_SABOTAGE: std::sync::Mutex<Option<DeviceRoleNameSabotage>> =
    std::sync::Mutex::new(None);

fn acl_input_error() -> ExecuteError {
    acl_device_failure("authorization candidate relation exceeds the bounded device input")
}

fn acl_device_failure(detail: impl Into<String>) -> ExecuteError {
    ExecuteError::Engine(EngineError::ApplyFailed(format!(
        "device authorization verdict failed closed: {}",
        detail.into()
    )))
}

fn table_privilege_code(privilege: TablePrivilege) -> u32 {
    match privilege {
        TablePrivilege::Select => ACL_PRIV_SELECT,
        TablePrivilege::Insert => ACL_PRIV_INSERT,
        TablePrivilege::Update => ACL_PRIV_UPDATE,
        TablePrivilege::Delete => ACL_PRIV_DELETE,
    }
}

fn function_privilege_code(privilege: FunctionPrivilege) -> u32 {
    match privilege {
        FunctionPrivilege::Execute => ACL_PRIV_SELECT,
    }
}

fn schema_privilege_code(privilege: SchemaPrivilege) -> u32 {
    match privilege {
        SchemaPrivilege::Usage => ACL_SCHEMA_USAGE,
        SchemaPrivilege::Create => ACL_SCHEMA_CREATE,
    }
}

fn device_acl_facts(catalog: &CatalogSnapshot) -> Result<DeviceAclFacts, ExecuteError> {
    let mut facts = DeviceAclFacts::default();
    // The public schema is an ordinary complete-generation candidate.  The kernel verifies its
    // kind and raw `public` name rather than treating an empty ACL as an implicit schema grant.
    if catalog.relational_public_schema_exists {
        facts.object(ACL_PUBLIC_SCHEMA_OID, ACL_TARGET_SCHEMA, "public")?;
    }
    // `postgres` is a modeled system role, not a host bypass.  It participates in the same
    // complete role-name resolution as SQL-visible roles before becoming the bootstrap principal.
    let (name_offset, name_len) = facts.text("postgres")?;
    facts.roles.push(DeviceAclRole {
        oid: ACL_BOOTSTRAP_OID,
        name_offset,
        name_len,
    });
    for role in catalog.relational_roles.values() {
        let (name_offset, name_len) = facts.text(&role.name)?;
        facts.roles.push(DeviceAclRole {
            oid: role.oid,
            name_offset,
            name_len,
        });
    }
    for table in catalog.relational_catalog.values() {
        facts.user_object(table.oid, ACL_TARGET_RELATION, &table.name)?;
        facts.table_grants(table.oid, ACL_TARGET_RELATION, &table.acl)?;
    }
    for view in catalog.relational_views.values() {
        facts.user_object(view.oid, ACL_TARGET_RELATION, &view.name)?;
        facts.table_grants(view.oid, ACL_TARGET_RELATION, &view.acl)?;
    }
    for view in catalog.relational_materialized_views.values() {
        facts.user_object(view.oid, ACL_TARGET_RELATION, &view.name)?;
        facts.table_grants(view.oid, ACL_TARGET_RELATION, &view.acl)?;
    }
    for sequence in catalog.relational_sequences.values() {
        facts.user_object(sequence.oid, ACL_TARGET_SEQUENCE, &sequence.name)?;
        facts.table_grants(sequence.oid, ACL_TARGET_SEQUENCE, &sequence.acl)?;
    }
    for function in catalog.relational_functions.values() {
        facts.user_object(function.oid, ACL_TARGET_FUNCTION, &function.name)?;
        facts.function_grants(function.oid, &function.acl)?;
    }
    // System policy is explicit raw candidate data.  The request remains a normal relation
    // request; the device matches these system objects and applies the policy only after target
    // resolution.  An unmodeled name returns MissingObject and the normal binder reports it.
    for (index, name) in SYSTEM_RELATION_CANDIDATES.iter().enumerate() {
        facts.object(
            u32::try_from(index)
                .map_err(|_| acl_input_error())?
                .saturating_add(1),
            if *name == "__bootstrap_catalog_control" {
                ACL_TARGET_CONTROL
            } else {
                ACL_TARGET_SYSTEM
            },
            name,
        )?;
    }
    for (grantee, privileges) in &catalog.relational_schema_acl {
        let (grantee_name_offset, grantee_name_len) = facts.text(grantee)?;
        let grantee_is_public = u32::from(grantee == "public");
        for privilege in privileges {
            facts.schema_grants.push(DeviceAclSchemaGrant {
                grantee_name_offset,
                grantee_name_len,
                privilege: schema_privilege_code(*privilege),
                grantee_is_public,
            });
        }
    }
    Ok(facts)
}

const SYSTEM_RELATION_CANDIDATES: &[&str] = &[
    "pg_catalog.pg_namespace",
    "pg_catalog.pg_class",
    "pg_catalog.pg_attribute",
    "pg_catalog.pg_type",
    "pg_catalog.pg_am",
    "pg_catalog.pg_roles",
    "pg_catalog.pg_tablespace",
    "pg_catalog.pg_constraint",
    "pg_catalog.pg_index",
    "pg_catalog.pg_attrdef",
    "pg_catalog.pg_description",
    "pg_catalog.pg_subscription",
    "pg_catalog.pg_extension",
    "pg_catalog.pg_default_acl",
    "pg_catalog.pg_indexes",
    "pg_catalog.pg_tables",
    "pg_catalog.pg_language",
    "pg_catalog.pg_proc",
    "pg_catalog.pg_views",
    "pg_catalog.pg_publication",
    "pg_catalog.pg_publication_rel",
    "pg_catalog.pg_publication_namespace",
    "pg_catalog.pg_publication_tables",
    "pg_catalog.pg_policy",
    "pg_catalog.pg_settings",
    "pg_catalog.pg_trigger",
    "pg_catalog.pg_statistic_ext",
    "pg_catalog.pg_inherits",
    "information_schema.tables",
    "information_schema.columns",
    "information_schema.schemata",
    "information_schema.table_constraints",
    "information_schema.key_column_usage",
    "information_schema.views",
    "pg_namespace",
    "pg_class",
    "pg_attribute",
    "pg_type",
    "pg_am",
    "pg_roles",
    "pg_tablespace",
    "pg_constraint",
    "pg_index",
    "pg_attrdef",
    "pg_description",
    "pg_subscription",
    "pg_extension",
    "pg_default_acl",
    "pg_indexes",
    "pg_tables",
    "pg_language",
    "pg_proc",
    "pg_views",
    "pg_publication",
    "pg_publication_rel",
    "pg_publication_namespace",
    "pg_publication_tables",
    "pg_policy",
    "pg_settings",
    "pg_trigger",
    "pg_statistic_ext",
    "pg_inherits",
    "__bootstrap_catalog_control",
];

fn authorize_command(
    engine: &Engine,
    catalog: &CatalogSnapshot,
    principal: AuthorizationPrincipal,
    command: &Command,
) -> Result<(), ExecuteError> {
    match command {
        Command::Insert(insert) => authorize_dml(
            engine,
            catalog,
            principal,
            &insert.table,
            ACL_PRIV_INSERT,
            !insert.returning.is_empty(),
        ),
        Command::Update(update) => authorize_dml(
            engine,
            catalog,
            principal,
            &update.table,
            ACL_PRIV_UPDATE,
            !update.returning.is_empty(),
        ),
        Command::Delete(delete) => authorize_dml(
            engine,
            catalog,
            principal,
            &delete.table,
            ACL_PRIV_DELETE,
            !delete.returning.is_empty(),
        ),
        Command::SequenceNextVal(nextval) => engine.authorize_device_target(
            catalog,
            principal,
            &nextval.name,
            AuthorizationTarget::Sequence,
            ACL_PRIV_UPDATE,
        ),
        Command::SequenceSetVal(setval) => engine.authorize_device_target(
            catalog,
            principal,
            &setval.name,
            AuthorizationTarget::Sequence,
            ACL_PRIV_UPDATE,
        ),
        Command::SequenceCurrVal(currval) => engine.authorize_device_target(
            catalog,
            principal,
            &currval.name,
            AuthorizationTarget::Sequence,
            ACL_PRIV_SELECT,
        ),
        Command::CreateView(view) => {
            engine.authorize_device_target(
                catalog,
                principal,
                "public",
                AuthorizationTarget::Schema,
                ACL_SCHEMA_CREATE,
            )?;
            engine.authorize_device_target(
                catalog,
                principal,
                &view.query.table,
                AuthorizationTarget::Relation,
                ACL_PRIV_SELECT,
            )
        }
        Command::CreateMaterializedView(view) => {
            engine.authorize_device_target(
                catalog,
                principal,
                "public",
                AuthorizationTarget::Schema,
                ACL_SCHEMA_CREATE,
            )?;
            engine.authorize_device_target(
                catalog,
                principal,
                &view.query.table,
                AuthorizationTarget::Relation,
                ACL_PRIV_SELECT,
            )
        }
        Command::CreateTable(_)
        | Command::CreateIndex(_)
        | Command::CreateSequence(_)
        | Command::CreateFunction(_)
        | Command::CreateDomain(_)
        | Command::CreatePublication(_)
        | Command::CreateSubscription(_) => engine.authorize_device_target(
            catalog,
            principal,
            "public",
            AuthorizationTarget::Schema,
            ACL_SCHEMA_CREATE,
        ),
        Command::Begin { .. }
        | Command::Commit { .. }
        | Command::Rollback { .. }
        | Command::ResetAll
        | Command::SetRole { .. }
        | Command::SessionControl { .. }
        | Command::ShowTransactionIsolation
        | Command::SelectLiteral(_)
        | Command::PreparedCatalog(_) => Ok(()),
        Command::Select(select) => engine.authorize_select_at(catalog, principal, select),
        Command::SelectFunction(call) => {
            engine.authorize_function_at(catalog, principal, &call.name)
        }
        // This bounded compatibility model intentionally has no role membership/ownership or
        // grant-option surface.  A virtual raw control object lets the terminal device policy
        // allow only the bootstrap principal without a host early-bypass.
        _ => engine.authorize_device_target(
            catalog,
            principal,
            "__bootstrap_catalog_control",
            AuthorizationTarget::Control,
            ACL_PRIV_SELECT,
        ),
    }
}

fn authorize_dml(
    engine: &Engine,
    catalog: &CatalogSnapshot,
    principal: AuthorizationPrincipal,
    table: &str,
    mutation_privilege: u32,
    has_returning: bool,
) -> Result<(), ExecuteError> {
    let target = table;
    engine.authorize_device_target(
        catalog,
        principal,
        target,
        AuthorizationTarget::Relation,
        mutation_privilege,
    )?;
    // PostgreSQL exposes returned tuple values as a relation read.  This second exact-snapshot
    // device verdict happens before admission/transaction claim/WAL and intentionally does not
    // change the frozen no-RETURNING UPDATE/DELETE predicate/assignment behavior.
    if has_returning {
        engine.authorize_device_target(
            catalog,
            principal,
            target,
            AuthorizationTarget::Relation,
            ACL_PRIV_SELECT,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(sql: &str) -> gpu_db_sql::ParsedCommand {
        gpu_db_sql::ParsedCommand::parse(sql).unwrap()
    }

    fn submit(engine: &Engine, txn_id: TxnId, sql: &str) {
        engine.submit_transaction(txn_id, parsed(sql)).unwrap();
    }

    #[test]
    fn role_oid_survives_rename_and_fails_closed_after_drop_recreate() {
        let engine = Engine::new_local();
        submit(&engine, 1, "CREATE ROLE acl_actor");
        submit(&engine, 2, "CREATE TABLE acl_target (id int4)");
        submit(&engine, 3, "GRANT SELECT ON TABLE acl_target TO acl_actor");
        let principal = engine.resolve_authorization_principal("acl_actor").unwrap();
        submit(&engine, 4, "ALTER ROLE acl_actor RENAME TO acl_renamed");
        engine
            .authorize_relation_for_session(None, principal, "acl_target", TablePrivilege::Select)
            .unwrap();
        submit(
            &engine,
            5,
            "REVOKE SELECT ON TABLE acl_target FROM acl_renamed",
        );
        submit(&engine, 6, "DROP ROLE acl_renamed");
        submit(&engine, 7, "CREATE ROLE acl_renamed");
        assert!(matches!(
            engine.authorize_relation_for_session(
                None,
                principal,
                "acl_target",
                TablePrivilege::Select
            ),
            Err(ExecuteError::PermissionDenied(_))
        ));
        assert!(matches!(
            engine.authorize_relation_for_session(
                None,
                principal,
                "pg_catalog.pg_class",
                TablePrivilege::Select
            ),
            Err(ExecuteError::PermissionDenied(_))
        ));
    }

    #[test]
    fn direct_and_public_grants_are_device_verdicts() {
        let engine = Engine::new_local();
        submit(&engine, 10, "CREATE ROLE direct_acl");
        submit(&engine, 11, "CREATE ROLE public_acl");
        submit(&engine, 12, "CREATE TABLE acl_grants (id int4)");
        submit(
            &engine,
            13,
            "GRANT SELECT ON TABLE acl_grants TO direct_acl",
        );
        submit(&engine, 14, "GRANT INSERT ON TABLE acl_grants TO PUBLIC");
        let direct = engine
            .resolve_authorization_principal("direct_acl")
            .unwrap();
        let public = engine
            .resolve_authorization_principal("public_acl")
            .unwrap();
        engine
            .authorize_relation_for_session(None, direct, "acl_grants", TablePrivilege::Select)
            .unwrap();
        engine
            .authorize_relation_for_session(None, public, "acl_grants", TablePrivilege::Insert)
            .unwrap();
        assert!(engine
            .authorize_relation_for_session(None, public, "acl_grants", TablePrivilege::Select)
            .is_err());
    }

    #[test]
    fn explicit_public_object_identity_avoids_system_shadow_but_bare_shadow_fails_closed() {
        let engine = Engine::new_local();
        submit(&engine, 17, "CREATE ROLE public_shadow_actor");
        submit(&engine, 18, "CREATE TABLE pg_class (id int4)");
        submit(
            &engine,
            19,
            "GRANT SELECT ON TABLE pg_class TO public_shadow_actor",
        );
        let principal = engine
            .resolve_authorization_principal("public_shadow_actor")
            .unwrap();
        engine
            .authorize_relation_for_session(
                None,
                principal,
                "public.pg_class",
                TablePrivilege::Select,
            )
            .unwrap();
        assert!(matches!(
            engine.authorize_relation_for_session(
                None,
                principal,
                "pg_class",
                TablePrivilege::Select,
            ),
            Err(ExecuteError::Engine(EngineError::ApplyFailed(_)))
        ));
        engine
            .authorize_relation_for_session(
                None,
                principal,
                "pg_catalog.pg_class",
                TablePrivilege::Select,
            )
            .unwrap();
    }

    #[test]
    fn same_named_public_base_table_keeps_mutation_route_while_bare_read_stays_ambiguous() {
        let engine = Engine::new_local();
        submit(&engine, 20, "CREATE ROLE pg_type_writer");
        submit(&engine, 21, "CREATE TABLE pg_type (id int4)");
        submit(
            &engine,
            22,
            "GRANT SELECT, INSERT ON TABLE pg_type TO pg_type_writer",
        );
        let principal = engine
            .resolve_authorization_principal("pg_type_writer")
            .unwrap();
        engine
            .authorize_relation_for_session(None, principal, "pg_type", TablePrivilege::Insert)
            .unwrap();
        engine
            .authorize_relation_for_session(
                None,
                principal,
                "public.pg_type",
                TablePrivilege::Select,
            )
            .unwrap();
        assert!(matches!(
            engine.authorize_relation_for_session(
                None,
                principal,
                "pg_type",
                TablePrivilege::Select
            ),
            Err(ExecuteError::Engine(EngineError::ApplyFailed(_)))
        ));
    }

    #[test]
    fn role_name_resolution_is_complete_candidate_device_work_and_fails_closed() {
        let engine = Engine::new_local();
        submit(&engine, 15, "CREATE ROLE device_actor");
        submit(&engine, 16, "CREATE ROLE device_decoy");
        assert!(matches!(
            engine.resolve_authorization_principal("device_actor"),
            Ok(AuthorizationPrincipal::RoleOid(_))
        ));
        assert!(matches!(
            engine.resolve_authorization_principal("missing_device_role"),
            Err(ExecuteError::UndefinedRole(role)) if role == "missing_device_role"
        ));

        // A duplicate in the full device candidate relation is malformed: no host iteration order
        // is allowed to choose which OID SQL-visible SET ROLE would receive.
        let catalog = engine.catalog_snapshot();
        let mut facts = device_acl_facts(&catalog).unwrap();
        let duplicate = *facts
            .roles
            .iter()
            .find(|candidate| {
                let start = candidate.name_offset as usize;
                let end = start + candidate.name_len as usize;
                facts.bytes.get(start..end) == Some(b"device_actor")
            })
            .unwrap();
        facts.roles.push(duplicate);
        let (name_offset, name_len) = facts.text("device_actor").unwrap();
        let anchor = catalog_relation_table(
            "pg_catalog",
            "__device_role_name_duplicate_test_anchor",
            &[("anchor", SqlType::Int4)],
        );
        let (_, memory) = engine
            .build_transient_relation_residency(&anchor, &[vec![SqlValue::Int4(0)]])
            .unwrap();
        assert_eq!(
            memory
                .catalog_role_name_verdict(
                    DeviceRoleNameRequest {
                        name_offset,
                        name_len,
                    },
                    &facts.roles,
                    &facts.bytes,
                )
                .unwrap(),
            DeviceRoleNameVerdict::InvalidInput
        );

        engine.sabotage_next_device_role_name_verdict("device_actor");
        assert!(matches!(
            engine.resolve_authorization_principal("device_actor"),
            Err(ExecuteError::Engine(EngineError::ApplyFailed(_)))
        ));
    }

    #[test]
    fn denied_returning_is_pre_wal_and_effect_free_for_every_dml_kind() {
        let engine = Engine::new_local();
        submit(&engine, 20, "CREATE ROLE returning_writer");
        submit(
            &engine,
            21,
            "CREATE TABLE returning_guard (id int4, v int4)",
        );
        submit(&engine, 22, "INSERT INTO returning_guard VALUES (1, 1)");
        submit(
            &engine,
            23,
            "GRANT INSERT, UPDATE, DELETE ON TABLE returning_guard TO returning_writer",
        );
        let principal = engine
            .resolve_authorization_principal("returning_writer")
            .unwrap();
        for (txn, sql) in [
            (24, "INSERT INTO returning_guard VALUES (2, 2) RETURNING id"),
            (
                25,
                "UPDATE returning_guard SET v = 2 WHERE id = 1 RETURNING v",
            ),
            (26, "DELETE FROM returning_guard WHERE id = 1 RETURNING id"),
        ] {
            let wal_before = engine.durable_wal_records().len();
            let error = engine
                .submit_transaction(
                    txn,
                    MutationRequest::new(parsed(sql)).with_principal(principal),
                )
                .unwrap_err();
            assert!(matches!(error, ExecuteError::PermissionDenied(_)));
            assert_eq!(engine.durable_wal_records().len(), wal_before);
        }
    }

    #[test]
    fn missing_targets_keep_normal_object_precedence_and_device_failure_is_closed() {
        let engine = Engine::new_local();
        submit(&engine, 30, "CREATE ROLE verdict_actor");
        let principal = engine
            .resolve_authorization_principal("verdict_actor")
            .unwrap();
        engine
            .authorize_relation_for_session(
                None,
                principal,
                "absent_acl_target",
                TablePrivilege::Select,
            )
            .unwrap();
        let wal_before = engine.durable_wal_records().len();
        engine.sabotage_next_device_acl_verdict("public");
        let error = engine
            .submit_transaction(
                31,
                MutationRequest::new(parsed("CREATE TABLE sabotage_acl (id int4)"))
                    .with_principal(principal),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ExecuteError::Engine(EngineError::ApplyFailed(_))
        ));
        assert_eq!(engine.durable_wal_records().len(), wal_before);
    }

    #[test]
    fn schema_function_sequence_and_bootstrap_policy_remain_device_gated() {
        let engine = Engine::new_local();
        submit(&engine, 40, "CREATE ROLE policy_actor");
        submit(&engine, 41, "CREATE SEQUENCE policy_seq");
        submit(
            &engine,
            42,
            "CREATE FUNCTION policy_fn() RETURNS int4 LANGUAGE sql AS 'SELECT 42'",
        );
        let principal = engine
            .resolve_authorization_principal("policy_actor")
            .unwrap();
        assert!(engine
            .authorize_function_at(&engine.catalog_snapshot(), principal, "policy_fn")
            .is_err());
        submit(&engine, 43, "GRANT USAGE ON SCHEMA public TO policy_actor");
        submit(
            &engine,
            44,
            "GRANT EXECUTE ON FUNCTION policy_fn() TO policy_actor",
        );
        submit(
            &engine,
            45,
            "GRANT UPDATE ON SEQUENCE policy_seq TO policy_actor",
        );
        engine
            .authorize_function_at(&engine.catalog_snapshot(), principal, "policy_fn")
            .unwrap();
        engine
            .authorize_device_target(
                &engine.catalog_snapshot(),
                principal,
                "policy_seq",
                AuthorizationTarget::Sequence,
                ACL_PRIV_UPDATE,
            )
            .unwrap();
        engine
            .authorize_relation_for_session(
                None,
                AuthorizationPrincipal::BootstrapPostgres,
                "missing_bootstrap_target",
                TablePrivilege::Select,
            )
            .unwrap();
    }

    #[test]
    fn public_schema_empty_acl_implies_usage_but_never_create_and_revoke_stays_closed() {
        let engine = Engine::new_local();
        submit(&engine, 46, "CREATE ROLE schema_creator");
        let principal = engine
            .resolve_authorization_principal("schema_creator")
            .unwrap();
        assert!(engine
            .authorize_device_target(
                &engine.catalog_snapshot(),
                principal,
                "public",
                AuthorizationTarget::Schema,
                ACL_SCHEMA_CREATE,
            )
            .is_err());
        submit(
            &engine,
            47,
            "GRANT CREATE ON SCHEMA public TO schema_creator",
        );
        engine
            .authorize_device_target(
                &engine.catalog_snapshot(),
                principal,
                "public",
                AuthorizationTarget::Schema,
                ACL_SCHEMA_CREATE,
            )
            .unwrap();
        submit(
            &engine,
            48,
            "REVOKE CREATE ON SCHEMA public FROM schema_creator",
        );
        assert!(engine
            .authorize_device_target(
                &engine.catalog_snapshot(),
                principal,
                "public",
                AuthorizationTarget::Schema,
                ACL_SCHEMA_CREATE,
            )
            .is_err());
    }

    #[test]
    fn missing_schema_usage_has_schema_precedence_over_object_privilege() {
        let engine = Engine::new_local();
        submit(&engine, 49, "CREATE ROLE schema_usage_actor");
        submit(&engine, 50, "CREATE ROLE schema_usage_other");
        submit(&engine, 51, "CREATE TABLE schema_usage_guard (id int4)");
        submit(
            &engine,
            52,
            "GRANT SELECT ON TABLE schema_usage_guard TO schema_usage_actor",
        );
        let principal = engine
            .resolve_authorization_principal("schema_usage_actor")
            .unwrap();
        submit(
            &engine,
            53,
            "GRANT USAGE ON SCHEMA public TO schema_usage_other",
        );
        let error = engine
            .authorize_relation_for_session(
                None,
                principal,
                "schema_usage_guard",
                TablePrivilege::Select,
            )
            .unwrap_err();
        assert!(
            matches!(error, ExecuteError::PermissionDenied(message) if message == "permission denied for schema")
        );
    }

    #[test]
    fn system_relation_candidates_cannot_authorize_same_named_sequences_or_functions() {
        let engine = Engine::new_local();
        submit(&engine, 54, "CREATE ROLE system_name_actor");
        submit(&engine, 55, "CREATE SEQUENCE pg_class");
        submit(
            &engine,
            56,
            "CREATE FUNCTION pg_class() RETURNS int4 LANGUAGE sql AS 'SELECT 42'",
        );
        let principal = engine
            .resolve_authorization_principal("system_name_actor")
            .unwrap();
        assert!(engine
            .authorize_device_target(
                &engine.catalog_snapshot(),
                principal,
                "pg_class",
                AuthorizationTarget::Sequence,
                ACL_PRIV_UPDATE,
            )
            .is_err());
        assert!(engine
            .authorize_device_target(
                &engine.catalog_snapshot(),
                principal,
                "pg_class",
                AuthorizationTarget::Function,
                ACL_PRIV_SELECT,
            )
            .is_err());
    }

    #[test]
    #[ignore = "requires a local NVIDIA driver and GPU"]
    fn device_acl_operator_reads_a_real_repeatable_read_pinned_generation() {
        let engine = Engine::new_local();
        submit(&engine, 60, "CREATE ROLE pinned_actor");
        submit(&engine, 61, "CREATE ROLE pinned_decoy");
        submit(&engine, 62, "CREATE TABLE pinned_acl_guard (id int4)");
        submit(
            &engine,
            63,
            "GRANT SELECT ON TABLE pinned_acl_guard TO pinned_actor",
        );
        let principal = engine
            .resolve_authorization_principal("pinned_actor")
            .unwrap();
        engine
            .authorize_relation_for_session(
                None,
                principal,
                "pinned_acl_guard",
                TablePrivilege::Select,
            )
            .unwrap();

        // A supported transactional catalog operation establishes the real repeatable-read
        // private generation.  Concurrent published ACL/role changes must not leak into it.
        submit(&engine, 64, "BEGIN ISOLATION LEVEL REPEATABLE READ");
        submit(
            &engine,
            64,
            "CREATE TABLE pinned_generation_stage (id int4)",
        );
        submit(
            &engine,
            65,
            "REVOKE SELECT ON TABLE pinned_acl_guard FROM pinned_actor",
        );
        engine
            .authorize_relation_for_session(
                Some(64),
                principal,
                "pinned_acl_guard",
                TablePrivilege::Select,
            )
            .unwrap();
        assert!(engine
            .authorize_relation_for_session(
                None,
                principal,
                "pinned_acl_guard",
                TablePrivilege::Select,
            )
            .is_err());

        submit(
            &engine,
            66,
            "ALTER ROLE pinned_actor RENAME TO pinned_actor_renamed",
        );
        assert!(matches!(
            engine.resolve_authorization_principal("pinned_actor"),
            Err(ExecuteError::UndefinedRole(role)) if role == "pinned_actor"
        ));
        assert!(matches!(
            engine.resolve_authorization_principal_at(Some(64), "pinned_actor"),
            Ok(AuthorizationPrincipal::RoleOid(_))
        ));
        submit(&engine, 67, "DROP ROLE pinned_actor_renamed");
        assert!(matches!(
            engine.resolve_authorization_principal("pinned_actor_renamed"),
            Err(ExecuteError::UndefinedRole(role)) if role == "pinned_actor_renamed"
        ));
        assert!(matches!(
            engine.resolve_authorization_principal_at(Some(64), "pinned_actor"),
            Ok(AuthorizationPrincipal::RoleOid(_))
        ));
        submit(&engine, 64, "ROLLBACK");
    }

    #[test]
    fn transactional_role_and_acl_ddl_are_rejected_pre_effect_until_their_wal_envelope_exists() {
        let engine = Engine::new_local();
        submit(&engine, 57, "CREATE ROLE transactional_acl_actor");
        submit(
            &engine,
            58,
            "CREATE TABLE transactional_acl_guard (id int4)",
        );
        let principal = engine
            .resolve_authorization_principal("transactional_acl_actor")
            .unwrap();
        for (txn_id, sql) in [
            (59, "CREATE ROLE transaction_only_role"),
            (
                60,
                "GRANT SELECT ON TABLE transactional_acl_guard TO transactional_acl_actor",
            ),
        ] {
            submit(&engine, txn_id, "BEGIN");
            let wal_before = engine.durable_wal_records().len();
            let error = engine.submit_transaction(txn_id, parsed(sql)).unwrap_err();
            assert!(matches!(error, ExecuteError::Unsupported(_)));
            assert_eq!(engine.durable_wal_records().len(), wal_before);
            submit(&engine, txn_id, "ROLLBACK");
        }
        assert!(engine
            .authorize_relation_for_session(
                None,
                principal,
                "transactional_acl_guard",
                TablePrivilege::Select,
            )
            .is_err());
        assert!(matches!(
            engine.resolve_authorization_principal("transaction_only_role"),
            Err(ExecuteError::UndefinedRole(role)) if role == "transaction_only_role"
        ));
    }
}
