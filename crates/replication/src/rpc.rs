use gpu_db_types::{EngineError, Index, LogEntry, Term};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVoteRequest {
    pub candidate_term: Term,
    pub candidate_id: u64,
    pub last_log_index: Index,
    pub last_log_term: Term,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestVoteResponse {
    pub granted: bool,
    pub voter_term: Term,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendEntriesRequest {
    pub leader_term: Term,
    pub prev_log_index: Index,
    pub prev_log_term: Term,
    pub entries: Vec<LogEntry>,
    pub leader_commit: Index,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendEntriesResponse {
    pub accepted: bool,
    pub follower_term: Term,
    pub follower_commit_index: Index,
    pub follower_applied_index: Index,
    pub error: Option<String>,
}

impl AppendEntriesRequest {
    pub fn encode_frame(&self) -> Vec<u8> {
        let mut out = Vec::new();
        write_u64(&mut out, self.leader_term);
        write_u64(&mut out, self.prev_log_index);
        write_u64(&mut out, self.prev_log_term);
        write_u64(&mut out, self.leader_commit);
        write_u64(&mut out, self.entries.len() as u64);
        for entry in &self.entries {
            write_u64(&mut out, entry.term);
            write_u64(&mut out, entry.index);
            write_bytes(&mut out, &entry.payload);
        }
        out
    }

    pub fn decode_frame(frame: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = FrameCursor::new(frame);
        let leader_term = cursor.read_u64("leader_term")?;
        let prev_log_index = cursor.read_u64("prev_log_index")?;
        let prev_log_term = cursor.read_u64("prev_log_term")?;
        let leader_commit = cursor.read_u64("leader_commit")?;
        let entry_count = cursor.read_len("entry_count")?;
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            entries.push(LogEntry {
                term: cursor.read_u64("entry.term")?,
                index: cursor.read_u64("entry.index")?,
                payload: cursor.read_bytes("entry.payload")?.to_vec().into(),
            });
        }
        cursor.finish()?;
        Ok(Self {
            leader_term,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit,
        })
    }
}

impl AppendEntriesResponse {
    pub fn encode_frame(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(u8::from(self.accepted));
        write_u64(&mut out, self.follower_term);
        write_u64(&mut out, self.follower_commit_index);
        write_u64(&mut out, self.follower_applied_index);
        match &self.error {
            Some(error) => {
                out.push(1);
                write_bytes(&mut out, error.as_bytes());
            }
            None => out.push(0),
        }
        out
    }

    pub fn decode_frame(frame: &[u8]) -> Result<Self, EngineError> {
        let mut cursor = FrameCursor::new(frame);
        let accepted = cursor.read_bool("accepted")?;
        let follower_term = cursor.read_u64("follower_term")?;
        let follower_commit_index = cursor.read_u64("follower_commit_index")?;
        let follower_applied_index = cursor.read_u64("follower_applied_index")?;
        let has_error = cursor.read_bool("has_error")?;
        let error = if has_error {
            let raw = cursor.read_bytes("error")?;
            Some(String::from_utf8(raw.to_vec()).map_err(|err| {
                EngineError::ProposalFailed(format!("invalid response error utf8: {err}"))
            })?)
        } else {
            None
        };
        cursor.finish()?;
        Ok(Self {
            accepted,
            follower_term,
            follower_commit_index,
            follower_applied_index,
            error,
        })
    }
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    write_u64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

struct FrameCursor<'a> {
    frame: &'a [u8],
    offset: usize,
}

impl<'a> FrameCursor<'a> {
    fn new(frame: &'a [u8]) -> Self {
        Self { frame, offset: 0 }
    }

    fn read_bool(&mut self, field: &'static str) -> Result<bool, EngineError> {
        let byte = *self
            .frame
            .get(self.offset)
            .ok_or_else(|| frame_error(format!("missing {field}")))?;
        self.offset += 1;
        match byte {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(frame_error(format!("invalid {field} bool byte {other}"))),
        }
    }

    fn read_u64(&mut self, field: &'static str) -> Result<u64, EngineError> {
        let bytes = self.read_exact(field, 8)?;
        Ok(u64::from_be_bytes(
            bytes
                .try_into()
                .expect("read_exact with len 8 should return 8 bytes"),
        ))
    }

    fn read_len(&mut self, field: &'static str) -> Result<usize, EngineError> {
        let raw = self.read_u64(field)?;
        usize::try_from(raw).map_err(|_| frame_error(format!("{field} length exceeds usize")))
    }

    fn read_bytes(&mut self, field: &'static str) -> Result<&'a [u8], EngineError> {
        let len = self.read_len(field)?;
        self.read_exact(field, len)
    }

    fn read_exact(&mut self, field: &'static str, len: usize) -> Result<&'a [u8], EngineError> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| frame_error(format!("{field} length overflows frame cursor")))?;
        if end > self.frame.len() {
            return Err(frame_error(format!(
                "{field} needs {len} bytes but frame has {} remaining",
                self.frame.len().saturating_sub(self.offset)
            )));
        }
        let bytes = &self.frame[self.offset..end];
        self.offset = end;
        Ok(bytes)
    }

    fn finish(&self) -> Result<(), EngineError> {
        if self.offset == self.frame.len() {
            Ok(())
        } else {
            Err(frame_error(format!(
                "frame has {} trailing bytes",
                self.frame.len() - self.offset
            )))
        }
    }
}

fn frame_error(message: String) -> EngineError {
    EngineError::ProposalFailed(format!("append entries frame decode failed: {message}"))
}
