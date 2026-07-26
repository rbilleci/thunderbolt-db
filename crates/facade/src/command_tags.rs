use super::CommandTag;
use gpu_db_sql::Command;

pub(super) fn reset_command_tag(source: &str) -> CommandTag {
    let normalized = source.trim().to_ascii_uppercase();
    let label = if normalized.starts_with("CLOSE ") {
        "CLOSE CURSOR"
    } else if normalized.starts_with("DISCARD ") {
        "DISCARD ALL"
    } else if normalized.starts_with("DEALLOCATE ") {
        "DEALLOCATE ALL"
    } else if normalized.starts_with("UNLISTEN") {
        "UNLISTEN"
    } else {
        "RESET"
    };
    CommandTag::Other(label.to_string())
}

pub(super) fn command_tag(command: &Command) -> CommandTag {
    match command {
        Command::CreateTable(_) => CommandTag::CreateTable,
        Command::CreateIndex(_) => CommandTag::CreateIndex,
        Command::Insert(_) => CommandTag::Insert,
        Command::Update(_) => CommandTag::Update,
        Command::Delete(_) => CommandTag::Delete,
        Command::TruncateTable(_) => CommandTag::Truncate,
        Command::Begin { .. } => CommandTag::Begin,
        Command::Commit { .. } => CommandTag::Commit,
        Command::Rollback { .. } => CommandTag::Rollback,
        Command::Flush => CommandTag::Other("CHECKPOINT".to_string()),
        Command::ResetAll => CommandTag::Other("RESET".to_string()),
        Command::SetRole { .. }
        | Command::SetKv { .. }
        | Command::DeleteKv { .. }
        | Command::SessionControl { .. } => CommandTag::Other("SET".to_string()),
        Command::GetKv { .. } | Command::ShowTransactionIsolation => {
            CommandTag::Other("SHOW".to_string())
        }
        Command::CreateSchema(_) => CommandTag::Other("CREATE SCHEMA".to_string()),
        Command::DropSchema(_) => CommandTag::Other("DROP SCHEMA".to_string()),
        Command::CreateDatabase(_) => CommandTag::Other("CREATE DATABASE".to_string()),
        Command::DropDatabase(_) => CommandTag::Other("DROP DATABASE".to_string()),
        Command::RenameDatabase(_) => CommandTag::Other("ALTER DATABASE".to_string()),
        Command::CreateTablespace(_) => CommandTag::Other("CREATE TABLESPACE".to_string()),
        Command::DropTablespace(_) => CommandTag::Other("DROP TABLESPACE".to_string()),
        Command::RenameTablespace(_) => CommandTag::Other("ALTER TABLESPACE".to_string()),
        Command::AddPrimaryKey(_)
        | Command::AddUniqueConstraint(_)
        | Command::AddCheckConstraint(_)
        | Command::AddForeignKey(_)
        | Command::AddColumn(_)
        | Command::RenameTable(_)
        | Command::RenameColumn(_)
        | Command::RenameConstraint(_)
        | Command::DropColumn(_)
        | Command::DropConstraint(_)
        | Command::AlterColumnDefault(_) => CommandTag::Other("ALTER TABLE".to_string()),
        Command::DropTable(_) => CommandTag::Other("DROP TABLE".to_string()),
        Command::DropIndex(_) => CommandTag::Other("DROP INDEX".to_string()),
        Command::RenameIndex(_) => CommandTag::Other("ALTER INDEX".to_string()),
        Command::CreateView(_) => CommandTag::Other("CREATE VIEW".to_string()),
        Command::RenameView(_) => CommandTag::Other("ALTER VIEW".to_string()),
        Command::DropView(_) => CommandTag::Other("DROP VIEW".to_string()),
        Command::CreateMaterializedView(_) => CommandTag::Other("SELECT 0".to_string()),
        Command::RefreshMaterializedView(_) => {
            CommandTag::Other("REFRESH MATERIALIZED VIEW".to_string())
        }
        Command::RenameMaterializedView(_) => {
            CommandTag::Other("ALTER MATERIALIZED VIEW".to_string())
        }
        Command::DropMaterializedView(_) => CommandTag::Other("DROP MATERIALIZED VIEW".to_string()),
        Command::CreateFunction(_) => CommandTag::Other("CREATE FUNCTION".to_string()),
        Command::RenameFunction(_) => CommandTag::Other("ALTER FUNCTION".to_string()),
        Command::DropFunction(_) => CommandTag::Other("DROP FUNCTION".to_string()),
        Command::CreateExtension(_) => CommandTag::Other("CREATE EXTENSION".to_string()),
        Command::DropExtension(_) => CommandTag::Other("DROP EXTENSION".to_string()),
        Command::CreateSequence(_) => CommandTag::Other("CREATE SEQUENCE".to_string()),
        Command::RenameSequence(_) | Command::SequenceRestart(_) => {
            CommandTag::Other("ALTER SEQUENCE".to_string())
        }
        Command::DropSequence(_) => CommandTag::Other("DROP SEQUENCE".to_string()),
        Command::CreateDomain(_) => CommandTag::Other("CREATE DOMAIN".to_string()),
        Command::DropDomain(_) => CommandTag::Other("DROP DOMAIN".to_string()),
        Command::CreatePublication(_) => CommandTag::Other("CREATE PUBLICATION".to_string()),
        Command::DropPublication(_) => CommandTag::Other("DROP PUBLICATION".to_string()),
        Command::CreateSubscription(_) => CommandTag::Other("CREATE SUBSCRIPTION".to_string()),
        Command::DropSubscription(_) => CommandTag::Other("DROP SUBSCRIPTION".to_string()),
        Command::CreateRole(_) => CommandTag::Other("CREATE ROLE".to_string()),
        Command::DropRole(_) => CommandTag::Other("DROP ROLE".to_string()),
        Command::RenameRole(_) | Command::AlterRoleLogin(_) => {
            CommandTag::Other("ALTER ROLE".to_string())
        }
        Command::GrantTable(_)
        | Command::GrantDatabase(_)
        | Command::GrantTablespace(_)
        | Command::GrantFunction(_)
        | Command::GrantSchema(_) => CommandTag::Other("GRANT".to_string()),
        Command::GrantDefaultTablePrivileges(_) => {
            CommandTag::Other("ALTER DEFAULT PRIVILEGES".to_string())
        }
        Command::RevokeTable(_)
        | Command::RevokeDatabase(_)
        | Command::RevokeTablespace(_)
        | Command::RevokeFunction(_)
        | Command::RevokeSchema(_) => CommandTag::Other("REVOKE".to_string()),
        Command::RevokeDefaultTablePrivileges(_) => {
            CommandTag::Other("ALTER DEFAULT PRIVILEGES".to_string())
        }
        Command::CommentOn(_) => CommandTag::Other("COMMENT".to_string()),
        Command::Select(_)
        | Command::SelectLiteral(_)
        | Command::SelectFunction(_)
        | Command::PreparedCatalog(_)
        | Command::SequenceNextVal(_)
        | Command::SequenceCurrVal(_)
        | Command::SequenceSetVal(_) => CommandTag::Other("SELECT".to_string()),
    }
}

pub(super) fn is_effect_free_session_function(call: &gpu_db_sql::SelectFunction) -> bool {
    call.name == "pg_advisory_unlock_all"
}
