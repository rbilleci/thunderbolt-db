use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use gpu_db_write_conveyor::{
    intent_for, recover_wal_segment_by_scan, stats_for_range, WriteIntent,
};

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
enum DirectMode {
    DataSync,
    FileData,
    RwfDataSync,
}

#[derive(Clone, Copy, Debug)]
struct Percentiles {
    p50: Duration,
    p90: Duration,
    p99: Duration,
    max: Duration,
    avg: Duration,
}

struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

impl AlignedBuffer {
    fn zeroed(len: usize, align: usize) -> io::Result<Self> {
        if len == 0 || !align.is_power_of_two() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "direct WAL buffer length must be non-zero and alignment must be a power of two",
            ));
        }
        let layout = Layout::from_size_align(len, align).map_err(io::Error::other)?;
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).ok_or_else(|| io::Error::other("direct WAL alloc failed"))?;
        Ok(Self { ptr, len, layout })
    }

    fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
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

fn parse_direct_modes() -> Result<Vec<DirectMode>, Box<dyn Error>> {
    let value =
        std::env::var("CONVEYOR_RAW_DIRECT_MODES").unwrap_or_else(|_| "direct-dsync".to_string());
    let mut modes = Vec::new();
    for item in value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
    {
        modes.push(match item {
            "direct-dsync" | "o-direct-o-dsync" => DirectMode::DataSync,
            "direct-fdatasync" | "o-direct-fdatasync" => DirectMode::FileData,
            "direct-rwf-dsync" | "o-direct-rwf-dsync" => DirectMode::RwfDataSync,
            value => {
                return Err(format!(
                    "unsupported CONVEYOR_RAW_DIRECT_MODES entry {value}; use direct-dsync, direct-fdatasync, or direct-rwf-dsync"
                )
                .into());
            }
        });
    }
    if modes.is_empty() {
        return Err("CONVEYOR_RAW_DIRECT_MODES must contain at least one mode".into());
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
        .unwrap_or(62)
        .max(1)
}

fn direct_mode_label(mode: DirectMode) -> &'static str {
    match mode {
        DirectMode::DataSync => "direct-dsync",
        DirectMode::FileData => "direct-fdatasync",
        DirectMode::RwfDataSync => "direct-rwf-dsync",
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

fn case_path(base: &Path, mode: DirectMode, fence_blocks: u64, multi_case: bool) -> PathBuf {
    if !multi_case {
        return base.to_path_buf();
    }
    let mut file_name = base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("raw-direct-wal")
        .to_string();
    file_name.push('.');
    file_name.push_str(direct_mode_label(mode));
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
    let events: u64 = parse_env("CONVEYOR_EVENTS", 200_000_u64);
    let block_size = parse_raw_block_size();
    let direct_align: usize = parse_env("CONVEYOR_RAW_DIRECT_ALIGN", 4096_usize).max(512);
    let groups = parse_group_list()?;
    let modes = parse_direct_modes()?;
    let queue_depth: usize = parse_env("CONVEYOR_RAW_DIRECT_QD", 1_usize).max(1);
    // Pre-write real bytes through a buffered descriptor so the timed region runs
    // over WRITTEN extents. posix_fallocate alone leaves XFS unwritten extents, and
    // then every durable fence pays an extent-conversion journal force (~3x fence
    // latency on this host). Production shape: recycled/pre-written segments.
    let prewrite = parse_bool("CONVEYOR_RAW_DIRECT_PREWRITE", true);
    let validate = parse_bool("CONVEYOR_RAW_VALIDATE", true);
    let keep_file = parse_bool("CONVEYOR_KEEP_FILE", false);
    let (base_path, generated_path) = match std::env::var("CONVEYOR_FILE") {
        Ok(path) => (PathBuf::from(path), false),
        Err(_) => (
            PathBuf::from(format!("target/raw-direct-wal-{}.dat", std::process::id())),
            true,
        ),
    };
    let overwrite = generated_path || parse_bool("CONVEYOR_OVERWRITE", false);

    if events == 0 {
        return Err("CONVEYOR_EVENTS must be greater than zero".into());
    }
    if events > usize::MAX as u64 {
        return Err("raw direct WAL event count exceeds usize".into());
    }

    let block_stride = block_stride(block_size)?;
    if !WAL_SEGMENT_HEADER_BYTES.is_multiple_of(direct_align)
        || !block_stride.is_multiple_of(direct_align)
    {
        return Err(format!(
            "direct WAL requires header and block stride aligned to {direct_align}B; block_size={block_size} gives stride={block_stride}. For 4096B alignment use CONVEYOR_RAW_WAL_BLOCK=62, 126, 190, ..."
        )
        .into());
    }
    let total_blocks = events.div_ceil(block_size as u64);
    if total_blocks > usize::MAX as u64 {
        return Err("raw direct WAL block count exceeds usize".into());
    }
    if total_blocks > u32::MAX as u64 {
        return Err("raw direct WAL block count must fit the WAL file header".into());
    }
    if block_size > u32::MAX as usize / std::mem::size_of::<WriteIntent>() {
        return Err("raw direct WAL block payload bytes must fit the WAL block header".into());
    }

    println!("raw direct WAL benchmark");
    println!(
        "  events={events} block_size={block_size} block_stride={block_stride} blocks={total_blocks} direct_align={direct_align}"
    );
    println!(
        "  direct_modes={} fence_blocks={} qd={queue_depth} prewrite={prewrite} validate={validate}",
        modes
            .iter()
            .map(|mode| direct_mode_label(*mode))
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
                direct_align,
                queue_depth,
                prewrite,
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
    mode: DirectMode,
    direct_align: usize,
    queue_depth: usize,
    prewrite: bool,
    validate: bool,
    keep_file: bool,
    overwrite: bool,
    path: &Path,
) -> Result<(), Box<dyn Error>> {
    remove_existing(path, overwrite)?;
    let setup_start = Instant::now();
    let file = create_direct_wal_file(path, mode, events, total_blocks, block_size, block_stride)?;
    if prewrite {
        prewrite_extents(
            path,
            WAL_SEGMENT_HEADER_BYTES as u64 + total_blocks * block_stride as u64,
        )?;
    }
    let setup = setup_start.elapsed();
    write_header(&file, mode, direct_align, total_blocks, block_size)?;

    let group_capacity = (fence_blocks as usize)
        .checked_mul(block_stride)
        .ok_or("raw direct WAL group buffer size overflow")?;
    let total_groups = total_blocks.div_ceil(fence_blocks);
    let mut publish_samples = Vec::with_capacity(total_blocks as usize);
    let mut fence_samples = Vec::with_capacity(total_groups as usize);
    let mut group_samples = Vec::with_capacity(total_groups as usize);

    // Fence lanes: each lane claims the next unfenced group and issues its own
    // durable write. FUA write-through fences (direct-dsync / direct-rwf-dsync)
    // are independent NVMe commands, so unlike a full-cache FLUSH they pipeline
    // across lanes; the device coalesces concurrent FUA writes internally.
    let group_cursor = AtomicU64::new(0);
    let errors: Mutex<Vec<io::Error>> = Mutex::new(Vec::new());
    let sample_sink: Mutex<(Vec<Duration>, Vec<Duration>, Vec<Duration>)> =
        Mutex::new((Vec::new(), Vec::new(), Vec::new()));
    let elapsed_start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..queue_depth {
            scope.spawn(|| {
                let mut group_buffer = match AlignedBuffer::zeroed(group_capacity, direct_align) {
                    Ok(buffer) => buffer,
                    Err(error) => {
                        errors.lock().expect("error sink poisoned").push(error);
                        return;
                    }
                };
                let mut publish_local = Vec::new();
                let mut fence_local = Vec::new();
                let mut group_local = Vec::new();
                loop {
                    let group = group_cursor.fetch_add(1, Ordering::Relaxed);
                    if group >= total_groups {
                        break;
                    }
                    let group_start = Instant::now();
                    let group_first_block = group * fence_blocks;
                    let group_end_blocks = group_first_block
                        .saturating_add(fence_blocks)
                        .min(total_blocks);
                    let group_block_count = (group_end_blocks - group_first_block) as usize;
                    for block_id in group_first_block..group_end_blocks {
                        let block_index = (block_id - group_first_block) as usize;
                        let first_seq = block_id * block_size as u64;
                        let count = (events - first_seq).min(block_size as u64) as usize;
                        let publish_start = Instant::now();
                        fill_block(
                            group_buffer.as_mut_slice(),
                            block_index * block_stride,
                            block_id,
                            first_seq,
                            count,
                            block_size,
                        );
                        publish_local.push(publish_start.elapsed());
                    }
                    let bytes = group_block_count * block_stride;
                    let file_offset =
                        WAL_SEGMENT_HEADER_BYTES as u64 + group_first_block * block_stride as u64;
                    let fence_start = Instant::now();
                    if let Err(error) = write_direct_group(
                        &file,
                        mode,
                        &group_buffer.as_slice()[..bytes],
                        file_offset,
                    ) {
                        errors.lock().expect("error sink poisoned").push(error);
                        return;
                    }
                    fence_local.push(fence_start.elapsed());
                    group_local.push(group_start.elapsed());
                }
                let mut sink = sample_sink.lock().expect("sample sink poisoned");
                sink.0.extend(publish_local);
                sink.1.extend(fence_local);
                sink.2.extend(group_local);
            });
        }
    });
    let elapsed = elapsed_start.elapsed();
    if let Some(error) = errors.into_inner().expect("error sink poisoned").pop() {
        return Err(error.into());
    }
    {
        let mut sink = sample_sink.into_inner().expect("sample sink poisoned");
        publish_samples.append(&mut sink.0);
        fence_samples.append(&mut sink.1);
        group_samples.append(&mut sink.2);
    }
    drop(file);
    println!(
        "  setup={:.3}s (create+fallocate{})",
        setup.as_secs_f64(),
        if prewrite { "+prewrite" } else { "" }
    );

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
                "raw direct WAL recovery mismatch: recovered_blocks={} recovered_records={} stats={:?} expected_blocks={total_blocks} expected_records={events} expected_stats={:?}",
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
        "  case direct_mode={} fence_blocks={} qd={queue_depth} prewrite={prewrite} elapsed={:.3}s throughput={:.3} M rec/s {:.3} M blocks/s {:.3} K fences/s",
        direct_mode_label(mode),
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

fn create_direct_wal_file(
    path: &Path,
    mode: DirectMode,
    events: u64,
    total_blocks: u64,
    block_size: usize,
    block_stride: usize,
) -> io::Result<File> {
    let mut flags = libc::O_NOFOLLOW | libc::O_DIRECT;
    if mode == DirectMode::DataSync {
        flags |= libc::O_DSYNC;
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .custom_flags(flags)
        .open(path)?;
    let bytes = WAL_SEGMENT_HEADER_BYTES as u64
        + total_blocks
            .checked_mul(block_stride as u64)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "direct WAL size overflow")
            })?;
    file.set_len(bytes)?;
    preallocate_file(&file, bytes)?;
    sync_parent_dir(path)?;
    let expected_capacity = (total_blocks as usize)
        .checked_mul(block_size)
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "direct WAL capacity overflow")
        })?;
    if expected_capacity < events as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "direct WAL capacity does not cover events",
        ));
    }
    Ok(file)
}

fn write_header(
    file: &File,
    mode: DirectMode,
    direct_align: usize,
    total_blocks: u64,
    block_size: usize,
) -> io::Result<()> {
    let mut header_buffer = AlignedBuffer::zeroed(WAL_SEGMENT_HEADER_BYTES, direct_align)?;
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
    header_buffer.as_mut_slice()[..std::mem::size_of::<RawWalSegmentFileHeader>()]
        .copy_from_slice(bytes_of(&header));
    write_direct_group(file, mode, header_buffer.as_slice(), 0)
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
    unsafe {
        let payload_ptr = buffer
            .as_mut_ptr()
            .add(payload_offset)
            .cast::<WriteIntent>();
        for offset in 0..count {
            payload_ptr
                .add(offset)
                .write(intent_for(first_client_seq + offset as u64));
        }
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

fn write_direct_group(file: &File, mode: DirectMode, bytes: &[u8], offset: u64) -> io::Result<()> {
    match mode {
        DirectMode::RwfDataSync => write_all_at_rwf_dsync(file, bytes, offset),
        DirectMode::DataSync => write_all_at(file, bytes, offset),
        DirectMode::FileData => {
            write_all_at(file, bytes, offset)?;
            file.sync_data()
        }
    }
}

fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        let written = file.write_at(bytes, offset)?;
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "direct WAL write returned zero",
            ));
        }
        bytes = &bytes[written..];
        offset += written as u64;
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn write_all_at_rwf_dsync(file: &File, mut bytes: &[u8], mut offset: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        let iov = libc::iovec {
            iov_base: bytes.as_ptr() as *mut libc::c_void,
            iov_len: bytes.len(),
        };
        let written = unsafe {
            libc::pwritev2(
                file.as_raw_fd(),
                &iov,
                1,
                offset as libc::off_t,
                libc::RWF_DSYNC,
            )
        };
        if written < 0 {
            return Err(io::Error::last_os_error());
        }
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "direct WAL pwritev2 returned zero",
            ));
        }
        let written = written as usize;
        bytes = &bytes[written..];
        offset += written as u64;
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn write_all_at_rwf_dsync(_file: &File, _bytes: &[u8], _offset: u64) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "direct-rwf-dsync requires Linux pwritev2(RWF_DSYNC)",
    ))
}

/// Convert the file's extents from fallocate-unwritten to written by streaming
/// real zeros through a buffered descriptor and fsyncing once. This runs at
/// setup time so hot-path durable fences never pay XFS unwritten-extent
/// conversion (a per-fence journal force). Production segments should be
/// recycled instead of recreated, amortizing this to zero.
fn prewrite_extents(path: &Path, bytes: u64) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?;
    let chunk = vec![0u8; 1 << 20];
    let mut remaining = bytes;
    while remaining > 0 {
        let step = remaining.min(chunk.len() as u64) as usize;
        file.write_all(&chunk[..step])?;
        remaining -= step as u64;
    }
    file.sync_all()?;
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
