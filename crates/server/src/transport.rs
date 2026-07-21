use std::io::{self, ErrorKind, Read, Write};

use tokio::io::{AsyncRead, AsyncReadExt};

/// PostgreSQL's declared message length includes the four-byte length field. Startup packets are
/// unauthenticated and intentionally receive a much tighter bound than ordinary/COPY messages.
pub(crate) const MAX_STARTUP_FRAME_BYTES: usize = 64 * 1024;
pub(crate) const MAX_AUTH_FRAME_BYTES: usize = 64 * 1024;
pub(crate) const MAX_TAGGED_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// The protocol/query layer is deliberately transport-neutral after startup. TLS wraps the socket
/// and then enters this same interface; it never introduces another dispatcher or engine path.
pub(crate) trait ReadWrite: Read + Write {}

impl<T: Read + Write> ReadWrite for T {}

pub(crate) fn read_startup_frame(stream: &mut dyn ReadWrite) -> io::Result<Option<Vec<u8>>> {
    let mut length = [0_u8; 4];
    if !read_first_byte(stream, &mut length[..1])? {
        return Ok(None);
    }
    stream.read_exact(&mut length[1..])?;
    let frame_len = validate_declared_frame_len(
        u32::from_be_bytes(length),
        8,
        MAX_STARTUP_FRAME_BYTES,
        "startup",
    )?;
    let mut frame = vec![0_u8; frame_len];
    frame[..4].copy_from_slice(&length);
    stream.read_exact(&mut frame[4..])?;
    Ok(Some(frame))
}

pub(crate) fn read_tagged_frame(stream: &mut dyn ReadWrite) -> io::Result<Option<Vec<u8>>> {
    read_tagged_frame_bounded(stream, MAX_TAGGED_FRAME_BYTES, "frontend")
}

/// SASL runs before authentication, so it does not inherit COPY's 64 MiB frame allowance.
pub(crate) fn read_auth_frame(stream: &mut dyn ReadWrite) -> io::Result<Option<Vec<u8>>> {
    read_tagged_frame_bounded(stream, MAX_AUTH_FRAME_BYTES, "authentication")
}

fn read_tagged_frame_bounded(
    stream: &mut dyn ReadWrite,
    maximum: usize,
    kind: &str,
) -> io::Result<Option<Vec<u8>>> {
    let mut tag = [0_u8; 1];
    if !read_first_byte(stream, &mut tag)? {
        return Ok(None);
    }
    let mut length = [0_u8; 4];
    stream.read_exact(&mut length)?;
    let frame_len = validate_declared_frame_len(u32::from_be_bytes(length), 4, maximum, kind)?;
    let mut frame = vec![0_u8; frame_len + 1];
    frame[0] = tag[0];
    frame[1..5].copy_from_slice(&length);
    stream.read_exact(&mut frame[5..])?;
    Ok(Some(frame))
}

fn read_first_byte(stream: &mut dyn ReadWrite, byte: &mut [u8]) -> io::Result<bool> {
    debug_assert_eq!(byte.len(), 1);
    loop {
        match stream.read(byte) {
            Ok(0) => return Ok(false),
            Ok(1) => return Ok(true),
            Ok(_) => unreachable!("one-byte read returned more than one byte"),
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
}

pub(crate) async fn read_startup_frame_async<R>(stream: &mut R) -> Result<Option<Vec<u8>>, String>
where
    R: AsyncRead + Unpin,
{
    let mut length = [0_u8; 4];
    if stream
        .read(&mut length[..1])
        .await
        .map_err(|error| error.to_string())?
        == 0
    {
        return Ok(None);
    }
    stream
        .read_exact(&mut length[1..])
        .await
        .map_err(|error| error.to_string())?;
    let frame_len = validate_declared_frame_len(
        u32::from_be_bytes(length),
        8,
        MAX_STARTUP_FRAME_BYTES,
        "startup",
    )
    .map_err(|error| error.to_string())?;
    let mut frame = vec![0_u8; frame_len];
    frame[..4].copy_from_slice(&length);
    stream
        .read_exact(&mut frame[4..])
        .await
        .map_err(|error| error.to_string())?;
    Ok(Some(frame))
}

pub(crate) async fn read_tagged_frame_async<R>(stream: &mut R) -> Result<Option<Vec<u8>>, String>
where
    R: AsyncRead + Unpin,
{
    let mut tag = [0_u8; 1];
    if stream
        .read(&mut tag)
        .await
        .map_err(|error| error.to_string())?
        == 0
    {
        return Ok(None);
    }
    let mut length = [0_u8; 4];
    stream
        .read_exact(&mut length)
        .await
        .map_err(|error| error.to_string())?;
    let frame_len = validate_declared_frame_len(
        u32::from_be_bytes(length),
        4,
        MAX_TAGGED_FRAME_BYTES,
        "frontend",
    )
    .map_err(|error| error.to_string())?;
    let mut frame = vec![0_u8; frame_len + 1];
    frame[0] = tag[0];
    frame[1..5].copy_from_slice(&length);
    stream
        .read_exact(&mut frame[5..])
        .await
        .map_err(|error| error.to_string())?;
    Ok(Some(frame))
}

pub(crate) fn validate_declared_frame_len(
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
    fn declared_lengths_accept_boundaries_and_reject_outside_them() {
        assert_eq!(
            validate_declared_frame_len(8, 8, MAX_STARTUP_FRAME_BYTES, "startup").unwrap(),
            8
        );
        assert_eq!(
            validate_declared_frame_len(
                MAX_STARTUP_FRAME_BYTES as u32,
                8,
                MAX_STARTUP_FRAME_BYTES,
                "startup",
            )
            .unwrap(),
            MAX_STARTUP_FRAME_BYTES
        );
        assert!(validate_declared_frame_len(7, 8, MAX_STARTUP_FRAME_BYTES, "startup").is_err());
        assert!(validate_declared_frame_len(
            MAX_STARTUP_FRAME_BYTES as u32 + 1,
            8,
            MAX_STARTUP_FRAME_BYTES,
            "startup",
        )
        .is_err());
        assert_eq!(
            validate_declared_frame_len(4, 4, MAX_TAGGED_FRAME_BYTES, "frontend").unwrap(),
            4
        );
    }

    #[test]
    fn blocking_readers_distinguish_clean_eof_from_partial_prefixes() {
        assert!(read_startup_frame(&mut Cursor::new(Vec::<u8>::new()))
            .unwrap()
            .is_none());
        assert_eq!(
            read_startup_frame(&mut Cursor::new(vec![0, 0]))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
        assert!(read_tagged_frame(&mut Cursor::new(Vec::<u8>::new()))
            .unwrap()
            .is_none());
        assert_eq!(
            read_tagged_frame(&mut Cursor::new(vec![b'Q', 0, 0]))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
    }

    #[test]
    fn blocking_readers_reject_bounds_before_payload_allocation() {
        let startup_over = (MAX_STARTUP_FRAME_BYTES as u32 + 1).to_be_bytes();
        assert_eq!(
            read_startup_frame(&mut Cursor::new(startup_over))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
        let tagged_over = [
            vec![b'Q'],
            (MAX_TAGGED_FRAME_BYTES as u32 + 1).to_be_bytes().to_vec(),
        ]
        .concat();
        assert_eq!(
            read_tagged_frame(&mut Cursor::new(tagged_over))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
        let auth_over = [
            vec![b'p'],
            (MAX_AUTH_FRAME_BYTES as u32 + 1).to_be_bytes().to_vec(),
        ]
        .concat();
        assert_eq!(
            read_auth_frame(&mut Cursor::new(auth_over))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
    }

    #[tokio::test]
    async fn async_readers_distinguish_clean_eof_and_apply_startup_bound() {
        let mut empty = &b""[..];
        assert!(read_startup_frame_async(&mut empty)
            .await
            .unwrap()
            .is_none());

        let mut partial = &b"\0\0"[..];
        assert!(read_startup_frame_async(&mut partial).await.is_err());

        let startup_over = (MAX_STARTUP_FRAME_BYTES as u32 + 1).to_be_bytes();
        let mut over = &startup_over[..];
        assert!(read_startup_frame_async(&mut over).await.is_err());
    }
}
