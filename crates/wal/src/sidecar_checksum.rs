//! Shared SHA-256 trailer codec for atomic WAL metadata sidecars.

use std::path::Path;

use gpu_db_types::EngineError;
use sha2::{Digest, Sha256};

pub(crate) fn append_sha256_trailer(mut body: String) -> String {
    debug_assert!(body.ends_with('\n'));
    let digest = Sha256::digest(body.as_bytes());
    body.push_str("sha256=");
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut body, "{byte:02x}").expect("writing to String cannot fail");
    }
    body.push('\n');
    body
}

pub(crate) fn verify_sha256_trailer(body: &str, path: &Path) -> Result<String, EngineError> {
    let trailer_start = body.rfind("sha256=").ok_or_else(|| {
        EngineError::Durability(format!(
            "checksummed durability artifact {} has no SHA-256 trailer",
            path.display()
        ))
    })?;
    if trailer_start > 0 && body.as_bytes()[trailer_start - 1] != b'\n' {
        return Err(EngineError::Durability(format!(
            "malformed SHA-256 trailer in {}",
            path.display()
        )));
    }
    let trailer = &body[trailer_start..];
    let Some(raw) = trailer.strip_prefix("sha256=") else {
        unreachable!();
    };
    let Some(raw) = raw.strip_suffix('\n') else {
        return Err(EngineError::Durability(format!(
            "unterminated SHA-256 trailer in {}",
            path.display()
        )));
    };
    if raw.len() != 64 || !raw.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(EngineError::Durability(format!(
            "invalid SHA-256 trailer in {}",
            path.display()
        )));
    }
    let protected = &body[..trailer_start];
    let digest = Sha256::digest(protected.as_bytes());
    let mut expected = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut expected, "{byte:02x}").expect("writing to String cannot fail");
    }
    if !raw.eq_ignore_ascii_case(&expected) {
        return Err(EngineError::Durability(format!(
            "SHA-256 checksum mismatch in {}",
            path.display()
        )));
    }
    Ok(protected.to_string())
}
