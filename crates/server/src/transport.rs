use std::future::Future;
use std::io::{self, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt};

/// PostgreSQL's declared message length includes the four-byte length field. Startup packets are
/// unauthenticated and intentionally receive a much tighter bound than ordinary/COPY messages.
pub(crate) const MAX_STARTUP_FRAME_BYTES: usize = 64 * 1024;
pub(crate) const MAX_AUTH_FRAME_BYTES: usize = 64 * 1024;
pub(crate) const MAX_TAGGED_FRAME_BYTES: usize = 64 * 1024 * 1024;
const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(25);

pub(crate) enum PolledTaggedFrame {
    Frame(Option<Vec<u8>>),
    Cancelled,
}

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
    read_tagged_frame_after_tag(stream, tag[0], maximum, kind).map(Some)
}

fn read_tagged_frame_after_tag(
    stream: &mut dyn ReadWrite,
    tag: u8,
    maximum: usize,
    kind: &str,
) -> io::Result<Vec<u8>> {
    let mut length = [0_u8; 4];
    stream.read_exact(&mut length)?;
    let frame_len = validate_declared_frame_len(u32::from_be_bytes(length), 4, maximum, kind)?;
    let mut frame = vec![0_u8; frame_len + 1];
    frame[0] = tag;
    frame[1..5].copy_from_slice(&length);
    stream.read_exact(&mut frame[5..])?;
    Ok(frame)
}

/// Wait for the first byte of a COPY-phase frame with a short socket timeout so a separate
/// CancelRequest connection can interrupt an otherwise idle COPY. Once a tag arrives, the timeout
/// is removed before reading the declared frame; a partial frame is therefore never discarded and
/// reparsed after a timeout.
pub(crate) fn read_tagged_frame_polling_cancel(
    stream: &mut dyn ReadWrite,
    timeout_control: &TcpStream,
    is_cancelled: impl Fn() -> bool,
) -> io::Result<PolledTaggedFrame> {
    timeout_control.set_read_timeout(Some(CANCELLATION_POLL_INTERVAL))?;
    let mut tag = [0_u8; 1];
    loop {
        match stream.read(&mut tag) {
            Ok(0) => {
                timeout_control.set_read_timeout(None)?;
                return Ok(PolledTaggedFrame::Frame(None));
            }
            Ok(1) => {
                timeout_control.set_read_timeout(None)?;
                return read_tagged_frame_after_tag(
                    stream,
                    tag[0],
                    MAX_TAGGED_FRAME_BYTES,
                    "frontend",
                )
                .map(|frame| PolledTaggedFrame::Frame(Some(frame)));
            }
            Ok(_) => unreachable!("one-byte read returned more than one byte"),
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                if is_cancelled() {
                    timeout_control.set_read_timeout(None)?;
                    return Ok(PolledTaggedFrame::Cancelled);
                }
            }
            Err(error) => {
                let _ = timeout_control.set_read_timeout(None);
                return Err(error);
            }
        }
    }
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
    read_tagged_frame_after_tag_async(stream, tag[0])
        .await
        .map(Some)
}

/// Wait cancellably only for the first byte of an async COPY-phase frame. An already-buffered tag
/// wins a simultaneous cancellation so the owner can drain that complete frame. Once a tag has
/// been consumed, finish and retain the exact frame before reporting cancellation to the caller;
/// this prevents a partial `read_exact` remainder becoming a new frontend frame during recovery.
pub(crate) async fn read_tagged_frame_polling_cancel_async<R, F>(
    stream: &mut R,
    cancelled: F,
) -> Result<PolledTaggedFrame, String>
where
    R: AsyncRead + Unpin,
    F: Future<Output = ()>,
{
    let mut tag = [0_u8; 1];
    let read = tokio::select! {
        biased;
        read = stream.read(&mut tag) => read.map_err(|error| error.to_string())?,
        () = cancelled => return Ok(PolledTaggedFrame::Cancelled),
    };
    if read == 0 {
        return Ok(PolledTaggedFrame::Frame(None));
    }
    read_tagged_frame_after_tag_async(stream, tag[0])
        .await
        .map(|frame| PolledTaggedFrame::Frame(Some(frame)))
}

async fn read_tagged_frame_after_tag_async<R>(stream: &mut R, tag: u8) -> Result<Vec<u8>, String>
where
    R: AsyncRead + Unpin,
{
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
    frame[0] = tag;
    frame[1..5].copy_from_slice(&length);
    stream
        .read_exact(&mut frame[5..])
        .await
        .map_err(|error| error.to_string())?;
    Ok(frame)
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
    use tokio::io::AsyncWriteExt;

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

    #[tokio::test]
    async fn async_copy_reader_cancels_before_a_tag_but_drains_a_started_frame() {
        let (_client, mut server) = tokio::io::duplex(64);
        assert!(matches!(
            read_tagged_frame_polling_cancel_async(&mut server, async {}).await,
            Ok(PolledTaggedFrame::Cancelled)
        ));

        let (mut client, mut server) = tokio::io::duplex(64);
        client.write_all(&[b'd', 0, 0, 0, 5, b'x']).await.unwrap();
        let result = read_tagged_frame_polling_cancel_async(&mut server, async {})
            .await
            .unwrap();
        let PolledTaggedFrame::Frame(Some(frame)) = result else {
            panic!("simultaneously ready complete COPY frame lost to cancellation");
        };
        assert_eq!(frame, vec![b'd', 0, 0, 0, 5, b'x']);

        let (mut client, mut server) = tokio::io::duplex(64);
        client.write_all(b"d").await.unwrap();
        let writer = tokio::spawn(async move {
            tokio::task::yield_now().await;
            client.write_all(&[0, 0, 0, 5, b'x']).await.unwrap();
        });
        // The cancellation future becomes ready after yielding once. Because the frame tag is
        // already buffered, the reader must commit to and drain that frame rather than drop a
        // partially-consumed read and desynchronize the next Sync.
        let result = read_tagged_frame_polling_cancel_async(&mut server, async {
            tokio::task::yield_now().await;
        })
        .await
        .unwrap();
        writer.await.unwrap();
        let PolledTaggedFrame::Frame(Some(frame)) = result else {
            panic!("started COPY frame was not retained");
        };
        assert_eq!(frame, vec![b'd', 0, 0, 0, 5, b'x']);
    }
}
