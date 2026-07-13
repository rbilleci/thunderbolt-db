use std::io::{self, ErrorKind, Read, Write};

/// PostgreSQL's declared message length includes the four-byte length field.
/// Keep ordinary/COPY compatibility while bounding unauthenticated allocation.
pub(super) const MAX_TAGGED_FRAME_BYTES: usize = 64 * 1024 * 1024;
pub(super) const MAX_STARTUP_FRAME_BYTES: usize = 64 * 1024;

pub(super) trait ReadWrite: Read + Write {}

impl<T: Read + Write> ReadWrite for T {}

pub(super) fn read_tagged_frame(stream: &mut dyn ReadWrite) -> io::Result<Option<Vec<u8>>> {
    let mut tag = [0_u8; 1];
    match stream.read_exact(&mut tag) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }

    let mut len_bytes = [0_u8; 4];
    stream.read_exact(&mut len_bytes)?;
    let frame_len = validate_declared_frame_len(
        u32::from_be_bytes(len_bytes),
        4,
        MAX_TAGGED_FRAME_BYTES,
        "tagged",
    )?;
    let mut frame = vec![0_u8; frame_len + 1];
    frame[0] = tag[0];
    frame[1..5].copy_from_slice(&len_bytes);
    stream.read_exact(&mut frame[5..])?;
    Ok(Some(frame))
}

pub(super) fn validate_declared_frame_len(
    declared: u32,
    minimum: usize,
    maximum: usize,
    kind: &str,
) -> io::Result<usize> {
    let declared = declared as usize;
    if declared < minimum {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("invalid {kind} frame length: {declared}"),
        ));
    }
    if declared > maximum {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("{kind} frame length {declared} exceeds limit {maximum}"),
        ));
    }
    Ok(declared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn declared_frame_lengths_accept_boundaries_and_reject_outside_them() {
        assert_eq!(
            validate_declared_frame_len(4, 4, MAX_TAGGED_FRAME_BYTES, "tagged").unwrap(),
            4
        );
        assert_eq!(
            validate_declared_frame_len(
                MAX_TAGGED_FRAME_BYTES as u32,
                4,
                MAX_TAGGED_FRAME_BYTES,
                "tagged",
            )
            .unwrap(),
            MAX_TAGGED_FRAME_BYTES
        );
        assert!(validate_declared_frame_len(3, 4, MAX_TAGGED_FRAME_BYTES, "tagged").is_err());
        assert!(validate_declared_frame_len(
            MAX_TAGGED_FRAME_BYTES as u32 + 1,
            4,
            MAX_TAGGED_FRAME_BYTES,
            "tagged",
        )
        .is_err());
    }

    #[test]
    fn tagged_frame_reconstructs_once_and_preserves_clean_eof() {
        assert!(read_tagged_frame(&mut Cursor::new(Vec::<u8>::new()))
            .unwrap()
            .is_none());
        let bytes = [vec![b'Q'], 7_u32.to_be_bytes().to_vec(), b"abc".to_vec()].concat();
        assert_eq!(
            read_tagged_frame(&mut Cursor::new(bytes.clone())).unwrap(),
            Some(bytes)
        );
    }

    #[test]
    fn tagged_frame_rejects_over_limit_before_payload_and_propagates_partial_reads() {
        let over = [
            vec![b'Q'],
            (MAX_TAGGED_FRAME_BYTES as u32 + 1).to_be_bytes().to_vec(),
        ]
        .concat();
        assert_eq!(
            read_tagged_frame(&mut Cursor::new(over))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(
            read_tagged_frame(&mut Cursor::new(vec![b'Q', 0, 0]))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
        let partial_payload = [vec![b'Q'], 7_u32.to_be_bytes().to_vec(), b"ab".to_vec()].concat();
        assert_eq!(
            read_tagged_frame(&mut Cursor::new(partial_payload))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
    }
}
