use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use gpu_db_write_conveyor::{
    intent_for, recover_wal_segment_by_scan, stats_for_range, WriteIntent,
};
use io_uring::{opcode, squeue, types, IoUring};

const WAL_SEGMENT_MAGIC: u64 = 0x5743_4f4e_5659_5347;
const WAL_BLOCK_HEADER_MAGIC: u64 = 0x5743_4f4e_4248_4452;
const WAL_BLOCK_TRAILER_MAGIC: u64 = 0x5743_4f4e_4254_524c;
const WAL_BLOCK_COMMIT_MARKER: u64 = 0x434f_4d4d_4954_4544;
const WAL_SEGMENT_VERSION: u32 = 1;
const WAL_SEGMENT_HEADER_BYTES: usize = 4096;

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
struct RawWalSegmentFileHeader {
    magic: u64,
    version: u32,
    header_bytes: u32,
    segment_id: u64,
    block_size: u32,
    block_capacity: u32,
    record_size: u32,
    block_header_size: u32,
    block_trailer_size: u32,
    flags: u32,
    reserved: [u64; 2],
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
struct RawWalBlockHeader {
    magic: u64,
    block_id: u64,
    first_client_seq: u64,
    count: u32,
    block_size: u32,
    payload_bytes: u32,
    reserved0: u32,
    reserved: [u64; 3],
}

#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default)]
struct RawWalBlockTrailer {
    magic: u64,
    block_id: u64,
    first_client_seq: u64,
    count: u32,
    payload_bytes: u32,
    checksum: u64,
    reserved: [u64; 2],
    commit_marker: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UringMode {
    RwfDataSync,
    WriteAndFileData,
    FixedRwfDataSync,
    FixedWriteAndFileData,
}

#[derive(Clone, Copy, Debug)]
struct Percentiles {
    p50: Duration,
    p90: Duration,
    p99: Duration,
    max: Duration,
    avg: Duration,
}

struct RegisteredBuffersGuard<'a> {
    ring: &'a mut IoUring<squeue::Entry>,
    registered: bool,
}

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

fn parse_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"),
        Err(_) => default,
    }
}

fn parse_uring_modes() -> Result<Vec<UringMode>, Box<dyn Error>> {
    let value =
        std::env::var("CONVEYOR_RAW_URING_MODES").unwrap_or_else(|_| "uring-rwf-dsync".to_string());
    let mut modes = Vec::new();
    for item in value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        modes.push(match item {
            "uring-rwf-dsync" | "uring-sync-write-data" => UringMode::RwfDataSync,
            "uring-write-fdatasync" | "uring-file-data" => UringMode::WriteAndFileData,
            "uring-fixed-rwf-dsync" | "uring-fixed-sync-write-data" => UringMode::FixedRwfDataSync,
            "uring-fixed-write-fdatasync" | "uring-fixed-file-data" => {
                UringMode::FixedWriteAndFileData
            }
            value => {
                return Err(format!(
                    "unsupported CONVEYOR_RAW_URING_MODES entry {value}; use uring-rwf-dsync, uring-write-fdatasync, uring-fixed-rwf-dsync, or uring-fixed-write-fdatasync"
                )
                .into());
            }
        });
    }
    if modes.is_empty() {
        return Err("CONVEYOR_RAW_URING_MODES must contain at least one mode".into());
    }
    Ok(modes)
}

fn parse_group_list() -> Result<Vec<u64>, Box<dyn Error>> {
    let value = std::env::var("CONVEYOR_RAW_FENCE_BLOCKS").unwrap_or_else(|_| "1".to_string());
    let mut groups = Vec::new();
    for item in value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        let group = item.parse::<u64>()?;
        if group == 0 {
            return Err("CONVEYOR_RAW_FENCE_BLOCKS entries must be greater than zero".into());
        }
        groups.push(group);
    }
    if groups.is_empty() {
        return Err("CONVEYOR_RAW_FENCE_BLOCKS must contain at least one value".into());
    }
    Ok(groups)
}

fn parse_raw_block_size() -> usize {
    std::env::var("CONVEYOR_RAW_WAL_BLOCK")
        .or_else(|_| std::env::var("CONVEYOR_CLIENT_WAL_BLOCK"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64)
        .max(1)
}

fn uring_mode_label(mode: UringMode) -> &'static str {
    match mode {
        UringMode::RwfDataSync => "uring-rwf-dsync",
        UringMode::WriteAndFileData => "uring-write-fdatasync",
        UringMode::FixedRwfDataSync => "uring-fixed-rwf-dsync",
        UringMode::FixedWriteAndFileData => "uring-fixed-write-fdatasync",
    }
}

impl UringMode {
    fn uses_fixed_buffer(self) -> bool {
        matches!(self, Self::FixedRwfDataSync | Self::FixedWriteAndFileData)
    }
}

fn percentile(sorted: &[Duration], percentile: usize) -> Duration {
    debug_assert!(!sorted.is_empty());
    sorted[(sorted.len() - 1) * percentile / 100]
}

fn distribution(samples: &[Duration]) -> Option<Percentiles> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let total_ns: u128 = sorted.iter().map(Duration::as_nanos).sum();
    Some(Percentiles {
        p50: percentile(&sorted, 50),
        p90: percentile(&sorted, 90),
        p99: percentile(&sorted, 99),
        max: *sorted.last().expect("non-empty sorted samples"),
        avg: Duration::from_nanos((total_ns / sorted.len() as u128) as u64),
    })
}

fn format_duration(duration: Duration) -> String {
    let ns = duration.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.2}us", ns as f64 / 1_000.0)
    } else {
        format!("{:.3}ms", ns as f64 / 1_000_000.0)
    }
}

fn print_distribution(label: &str, samples: &[Duration]) {
    match distribution(samples) {
        Some(dist) => println!(
            "    {label:<22} avg={} p50={} p90={} p99={} max={} samples={}",
            format_duration(dist.avg),
            format_duration(dist.p50),
            format_duration(dist.p90),
            format_duration(dist.p99),
            format_duration(dist.max),
            samples.len()
        ),
        None => println!("    {label:<22} no samples"),
    }
}

impl<'a> RegisteredBuffersGuard<'a> {
    fn new(ring: &'a mut IoUring, registered: bool) -> Self {
        Self { ring, registered }
    }

    fn ring(&mut self) -> &mut IoUring {
        self.ring
    }

    fn unregister(&mut self) -> io::Result<()> {
        if self.registered {
            self.ring.submitter().unregister_buffers()?;
            self.registered = false;
        }
        Ok(())
    }
}

impl Drop for RegisteredBuffersGuard<'_> {
    fn drop(&mut self) {
        if self.registered {
            let _ = self.ring.submitter().unregister_buffers();
        }
    }
}

fn case_path(base: &Path, mode: UringMode, fence_blocks: u64, multi_case: bool) -> PathBuf {
    if !multi_case {
        return base.to_path_buf();
    }
    let mut file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("raw-uring-wal")
        .to_string();
    file_name.push('.');
    file_name.push_str(uring_mode_label(mode));
    file_name.push('.');
    file_name.push_str(&fence_blocks.to_string());
    file_name.push_str(".dat");
    base.with_file_name(file_name)
}

fn remove_existing(path: &Path, overwrite: bool) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        if !overwrite {
            return Err(format!(
                "{} already exists; set CONVEYOR_OVERWRITE=1 to replace it",
                path.display()
            )
            .into());
        }
        std::fs::remove_file(path)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let events: u64 = parse_env("CONVEYOR_EVENTS", 500_000_u64);
    let block_size = parse_raw_block_size();
    let groups = parse_group_list()?;
    let modes = parse_uring_modes()?;
    let ring_entries: u32 = parse_env("CONVEYOR_RAW_URING_ENTRIES", 8_u32).max(2);
    let validate = parse_bool("CONVEYOR_RAW_VALIDATE", true);
    let keep_file = parse_bool("CONVEYOR_KEEP_FILE", false);
    let (base_path, generated_path) = match std::env::var("CONVEYOR_FILE") {
        Ok(path) => (PathBuf::from(path), false),
        Err(_) => (
            PathBuf::from(format!("target/raw-uring-wal-{}.dat", std::process::id())),
            true,
        ),
    };
    let overwrite = generated_path || parse_bool("CONVEYOR_OVERWRITE", false);

    if events == 0 {
        return Err("CONVEYOR_EVENTS must be greater than zero".into());
    }
    if events > usize::MAX as u64 {
        return Err("raw io_uring WAL event count exceeds usize".into());
    }
    let block_stride = block_stride(block_size)?;
    let total_blocks = events.div_ceil(block_size as u64);
    if total_blocks > usize::MAX as u64 {
        return Err("raw io_uring WAL block count exceeds usize".into());
    }
    if total_blocks > u32::MAX as u64 {
        return Err("raw io_uring WAL block count must fit the WAL file header".into());
    }
    if block_size > u32::MAX as usize / std::mem::size_of::<WriteIntent>() {
        return Err("raw io_uring WAL block payload bytes must fit the WAL block header".into());
    }

    println!("raw io_uring WAL benchmark");
    println!(
        "  events={events} block_size={block_size} block_stride={block_stride} blocks={total_blocks} ring_entries={ring_entries}"
    );
    println!(
        "  uring_modes={} fence_blocks={} validate={validate}",
        modes
            .iter()
            .map(|mode| uring_mode_label(*mode))
            .collect::<Vec<_>>()
            .join(","),
        groups
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",")
    );

    let multi_case = modes.len() > 1 || groups.len() > 1;
    for mode in modes {
        for &fence_blocks in groups.iter() {
            run_case(
                events,
                total_blocks,
                block_size,
                block_stride,
                fence_blocks,
                mode,
                ring_entries,
                validate,
                keep_file,
                overwrite,
                &case_path(&base_path, mode, fence_blocks, multi_case),
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    events: u64,
    total_blocks: u64,
    block_size: usize,
    block_stride: usize,
    fence_blocks: u64,
    mode: UringMode,
    ring_entries: u32,
    validate: bool,
    keep_file: bool,
    overwrite: bool,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    remove_existing(path, overwrite)?;
    let file = create_wal_file(path, events, total_blocks, block_size, block_stride)?;
    write_header(&file, total_blocks, block_size)?;
    let mut ring = IoUring::new(ring_entries)?;

    let group_capacity = (fence_blocks as usize)
        .checked_mul(block_stride)
        .ok_or("raw io_uring WAL group buffer size overflow")?;
    let mut group_buffer = vec![0_u8; group_capacity];
    let registered_buffers = mode.uses_fixed_buffer();
    if registered_buffers {
        let iovec = libc::iovec {
            iov_base: group_buffer.as_mut_ptr().cast::<libc::c_void>(),
            iov_len: group_buffer.len(),
        };
        unsafe {
            ring.submitter().register_buffers(&[iovec])?;
        }
    }
    let mut ring_guard = RegisteredBuffersGuard::new(&mut ring, registered_buffers);
    let mut publish_samples = Vec::with_capacity(total_blocks as usize);
    let mut fence_samples = Vec::with_capacity(total_blocks.div_ceil(fence_blocks) as usize);
    let mut group_samples = Vec::with_capacity(total_blocks.div_ceil(fence_blocks) as usize);
    let mut published_records = 0_u64;
    let mut published_blocks = 0_u64;

    let elapsed_start = Instant::now();
    while published_blocks < total_blocks {
        let group_start = Instant::now();
        let group_first_block = published_blocks;
        let group_end_blocks = published_blocks
            .saturating_add(fence_blocks)
            .min(total_blocks);
        let group_block_count = (group_end_blocks - group_first_block) as usize;
        while published_blocks < group_end_blocks {
            let block_index = (published_blocks - group_first_block) as usize;
            let remaining_records = events - published_records;
            let count = remaining_records.min(block_size as u64) as usize;
            let publish_start = Instant::now();
            fill_block(
                &mut group_buffer,
                block_index * block_stride,
                published_blocks,
                published_records,
                count,
                block_size,
            );
            publish_samples.push(publish_start.elapsed());
            published_records += count as u64;
            published_blocks += 1;
        }

        let bytes = group_block_count * block_stride;
        let file_offset = WAL_SEGMENT_HEADER_BYTES as u64 + group_first_block * block_stride as u64;
        let fence_start = Instant::now();
        write_uring_group(
            ring_guard.ring(),
            &file,
            mode,
            &group_buffer[..bytes],
            file_offset,
        )?;
        fence_samples.push(fence_start.elapsed());
        group_samples.push(group_start.elapsed());
    }
    let elapsed = elapsed_start.elapsed();
    ring_guard.unregister()?;
    drop(ring_guard);
    drop(file);

    let recovery_start = Instant::now();
    let recovery_result = if validate {
        Some((recover_wal_segment_by_scan(path)?, recovery_start.elapsed()))
    } else {
        None
    };

    if let Some((recovered, _recovery_elapsed)) = &recovery_result {
        let expected = stats_for_range(0, events);
        if recovered.recovered_blocks != total_blocks
            || recovered.recovered_records != events
            || recovered.stats != expected
        {
            return Err(format!(
                "raw io_uring WAL recovery mismatch: recovered_blocks={} recovered_records={} stats={:?} expected_blocks={total_blocks} expected_records={events} expected_stats={:?}",
                recovered.recovered_blocks, recovered.recovered_records, recovered.stats, expected
            )
            .into());
        }
    }

    if !keep_file {
        std::fs::remove_file(path)?;
    }

    let records_per_sec = events as f64 / elapsed.as_secs_f64();
    let blocks_per_sec = total_blocks as f64 / elapsed.as_secs_f64();
    let fences_per_sec = fence_samples.len() as f64 / elapsed.as_secs_f64();
    println!(
        "  case uring_mode={} fence_blocks={} elapsed={:.3}s throughput={:.3} M rec/s {:.3} M blocks/s {:.3} K fences/s",
        uring_mode_label(mode),
        fence_blocks,
        elapsed.as_secs_f64(),
        records_per_sec / 1_000_000.0,
        blocks_per_sec / 1_000_000.0,
        fences_per_sec / 1_000.0,
    );
    print_distribution("publish/block", &publish_samples);
    print_distribution("fence/group", &fence_samples);
    print_distribution("publish+fence/group", &group_samples);
    match recovery_result {
        Some((_recovered, recovery_elapsed)) => {
            println!(
                "    recovery/scan          {}",
                format_duration(recovery_elapsed)
            );
        }
        None => println!("    recovery/scan          skipped"),
    }
    Ok(())
}

fn create_wal_file(
    path: &Path,
    events: u64,
    total_blocks: u64,
    block_size: usize,
    block_stride: usize,
) -> io::Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let bytes = WAL_SEGMENT_HEADER_BYTES as u64
        + total_blocks
            .checked_mul(block_stride as u64)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "io_uring WAL size overflow")
            })?;
    file.set_len(bytes)?;
    preallocate_file(&file, bytes)?;
    sync_parent_dir(path)?;
    let expected_capacity = (total_blocks as usize)
        .checked_mul(block_size)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "io_uring WAL capacity overflow",
            )
        })?;
    if expected_capacity < events as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "io_uring WAL capacity does not cover events",
        ));
    }
    Ok(file)
}

fn write_header(file: &File, total_blocks: u64, block_size: usize) -> io::Result<()> {
    let mut header_bytes = vec![0_u8; WAL_SEGMENT_HEADER_BYTES];
    let header = RawWalSegmentFileHeader {
        magic: WAL_SEGMENT_MAGIC,
        version: WAL_SEGMENT_VERSION,
        header_bytes: WAL_SEGMENT_HEADER_BYTES as u32,
        segment_id: 0,
        block_size: block_size as u32,
        block_capacity: total_blocks as u32,
        record_size: size_of_u32::<WriteIntent>()?,
        block_header_size: size_of_u32::<RawWalBlockHeader>()?,
        block_trailer_size: size_of_u32::<RawWalBlockTrailer>()?,
        flags: 0,
        reserved: [0; 2],
    };
    header_bytes[..std::mem::size_of::<RawWalSegmentFileHeader>()]
        .copy_from_slice(bytes_of(&header));
    write_all_at(file, &header_bytes, 0)?;
    file.sync_data()
}

fn fill_block(
    buffer: &mut [u8],
    block_offset: usize,
    block_id: u64,
    first_client_seq: u64,
    count: usize,
    block_size: usize,
) {
    let payload_bytes = count * std::mem::size_of::<WriteIntent>();
    let header = RawWalBlockHeader {
        magic: WAL_BLOCK_HEADER_MAGIC,
        block_id,
        first_client_seq,
        count: count as u32,
        block_size: block_size as u32,
        payload_bytes: payload_bytes as u32,
        reserved0: 0,
        reserved: [0; 3],
    };
    let header_offset = block_offset;
    let payload_offset = header_offset + std::mem::size_of::<RawWalBlockHeader>();
    let trailer_offset = payload_offset + block_size * std::mem::size_of::<WriteIntent>();
    buffer[header_offset..payload_offset].copy_from_slice(bytes_of(&header));
    for offset in 0..count {
        let intent = intent_for(first_client_seq + offset as u64);
        let start = payload_offset + offset * std::mem::size_of::<WriteIntent>();
        let end = start + std::mem::size_of::<WriteIntent>();
        buffer[start..end].copy_from_slice(bytes_of(&intent));
    }
    if count < block_size {
        let unused_start = payload_offset + payload_bytes;
        let unused_end = trailer_offset;
        buffer[unused_start..unused_end].fill(0);
    }
    let checksum = block_crc32c(
        &header,
        &buffer[payload_offset..payload_offset + payload_bytes],
    );
    let trailer = RawWalBlockTrailer {
        magic: WAL_BLOCK_TRAILER_MAGIC,
        block_id,
        first_client_seq,
        count: count as u32,
        payload_bytes: payload_bytes as u32,
        checksum: checksum as u64,
        reserved: [0; 2],
        commit_marker: WAL_BLOCK_COMMIT_MARKER,
    };
    buffer[trailer_offset..trailer_offset + std::mem::size_of::<RawWalBlockTrailer>()]
        .copy_from_slice(bytes_of(&trailer));
}

fn write_uring_group(
    ring: &mut IoUring,
    file: &File,
    mode: UringMode,
    bytes: &[u8],
    offset: u64,
) -> io::Result<()> {
    match mode {
        UringMode::RwfDataSync => write_all_at_uring(ring, file, bytes, offset, libc::RWF_DSYNC),
        UringMode::WriteAndFileData => {
            write_all_at_uring(ring, file, bytes, offset, 0)?;
            file.sync_data()
        }
        UringMode::FixedRwfDataSync => {
            write_all_at_uring_fixed(ring, file, bytes, offset, libc::RWF_DSYNC)
        }
        UringMode::FixedWriteAndFileData => {
            write_all_at_uring_fixed(ring, file, bytes, offset, 0)?;
            file.sync_data()
        }
    }
}

fn write_all_at_uring(
    ring: &mut IoUring,
    file: &File,
    mut bytes: &[u8],
    mut offset: u64,
    rw_flags: i32,
) -> io::Result<()> {
    while !bytes.is_empty() {
        let len = bytes.len().min(u32::MAX as usize);
        let entry = opcode::Write::new(types::Fd(file.as_raw_fd()), bytes.as_ptr(), len as u32)
            .offset(offset)
            .rw_flags(rw_flags)
            .build()
            .user_data(offset);
        unsafe {
            ring.submission()
                .push(&entry)
                .map_err(|_| io::Error::other("io_uring submission queue is full"))?;
        }
        ring.submit_and_wait(1)?;
        let completion = ring
            .completion()
            .next()
            .ok_or_else(|| io::Error::other("io_uring completion queue is empty"))?;
        let result = completion.result();
        if result < 0 {
            return Err(io::Error::from_raw_os_error(-result));
        }
        if result == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "io_uring WAL write returned zero",
            ));
        }
        let written = result as usize;
        bytes = &bytes[written..];
        offset += written as u64;
    }
    Ok(())
}

fn write_all_at_uring_fixed(
    ring: &mut IoUring,
    file: &File,
    bytes: &[u8],
    offset: u64,
    rw_flags: i32,
) -> io::Result<()> {
    let len = u32::try_from(bytes.len()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "io_uring fixed WAL write exceeds u32",
        )
    })?;
    let entry = opcode::WriteFixed::new(types::Fd(file.as_raw_fd()), bytes.as_ptr(), len, 0)
        .offset(offset)
        .rw_flags(rw_flags)
        .build()
        .user_data(offset);
    unsafe {
        ring.submission()
            .push(&entry)
            .map_err(|_| io::Error::other("io_uring submission queue is full"))?;
    }
    ring.submit_and_wait(1)?;
    let completion = ring
        .completion()
        .next()
        .ok_or_else(|| io::Error::other("io_uring completion queue is empty"))?;
    let result = completion.result();
    if result < 0 {
        return Err(io::Error::from_raw_os_error(-result));
    }
    if result as usize != bytes.len() {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            format!(
                "io_uring fixed WAL write completed {result} bytes, expected {}",
                bytes.len()
            ),
        ));
    }
    Ok(())
}

fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        let written = file.write_at(bytes, offset)?;
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "io_uring WAL header write returned zero",
            ));
        }
        bytes = &bytes[written..];
        offset += written as u64;
    }
    Ok(())
}

fn preallocate_file(file: &File, bytes: u64) -> io::Result<()> {
    let rc = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, bytes as libc::off_t) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(rc))
    }
}

fn sync_parent_dir(path: &Path) -> io::Result<()> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let dir = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(parent)?;
    dir.sync_all()
}

fn block_stride(block_size: usize) -> io::Result<usize> {
    std::mem::size_of::<RawWalBlockHeader>()
        .checked_add(
            block_size
                .checked_mul(std::mem::size_of::<WriteIntent>())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "payload overflow"))?,
        )
        .and_then(|value| value.checked_add(std::mem::size_of::<RawWalBlockTrailer>()))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "block stride overflow"))
}

fn block_crc32c(header: &RawWalBlockHeader, payload: &[u8]) -> u32 {
    let crc = crc32c::crc32c_append(0, bytes_of(header));
    crc32c::crc32c_append(crc, payload)
}

fn bytes_of<T>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts((value as *const T).cast::<u8>(), std::mem::size_of::<T>())
    }
}

fn size_of_u32<T>() -> io::Result<u32> {
    u32::try_from(std::mem::size_of::<T>())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "size does not fit u32"))
}
