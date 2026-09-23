//! Random-access weight I/O for GGUF streaming (PRD §3.1).
//!
//! Linux production: `io_uring` Read at absolute offsets.
//! Windows / fallback: blocking `File` seek+read (dev host).

use super::IoBackend;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use tracing::debug;

/// Deterministic weight reads into caller buffers (no mmap of payloads).
pub trait WeightIo: Send {
    fn path(&self) -> &Path;
    fn backend(&self) -> IoBackend;
    /// Total size of the backing file in bytes. Used to reject GGUF metadata
    /// that would otherwise drive huge allocations/bounds checks from untrusted
    /// values.
    fn file_size(&self) -> u64;
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Batch reads into disjoint regions of `dst` (Phase E: `io_uring` submits all
    /// requests at once and waits once, instead of one syscall wait per tensor).
    /// Default implementation falls back to sequential [`Self::read_at`].
    fn read_many_at(&mut self, dst: &mut [u8], requests: &[IoRange]) -> io::Result<()> {
        validate_ranges(dst.len(), requests)?;
        for r in requests {
            self.read_at(r.offset, &mut dst[r.start..r.end])?;
        }
        Ok(())
    }
}

/// Validate that every [`IoRange`] lies inside `dst` and that ranges are
/// disjoint. io_uring writes through raw pointers computed from these ranges, so
/// an out-of-bounds or overlapping request would be UB. Disjointness is checked
/// in `O(n log n)` by sorting on the start offset.
fn validate_ranges(dst_len: usize, requests: &[IoRange]) -> io::Result<()> {
    for r in requests {
        if r.start > r.end || r.end > dst_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "io range {}..{} out of bounds for buffer of {}",
                    r.start, r.end, dst_len
                ),
            ));
        }
    }
    let mut order: Vec<usize> = (0..requests.len()).collect();
    order.sort_unstable_by_key(|&i| requests[i].start);
    for w in order.windows(2) {
        if requests[w[0]].end > requests[w[1]].start {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "overlapping io ranges",
            ));
        }
    }
    Ok(())
}

/// One batched read: absolute file `offset` into `dst[start..end]`.
#[derive(Debug, Clone, Copy)]
pub struct IoRange {
    pub offset: u64,
    pub start: usize,
    pub end: usize,
}

/// Open the preferred weight reader (`io_uring` on Linux unless `HAYAI_FORCE_TOKIO_IO=1`).
pub fn open_weight_io<P: AsRef<Path>>(path: P) -> io::Result<Box<dyn WeightIo>> {
    let path = path.as_ref();
    #[cfg(target_os = "linux")]
    {
        let force_stdio = std::env::var_os("HAYAI_FORCE_TOKIO_IO").is_some();
        if !force_stdio {
            match IoUringWeightIo::open(path) {
                Ok(r) => return Ok(Box::new(r)),
                Err(e) => {
                    tracing::warn!("io_uring WeightIo open failed ({e}); falling back to File I/O");
                }
            }
        }
    }
    Ok(Box::new(FileWeightIo::open(path)?))
}

/// Blocking file seek+read (Windows dev host + Linux fallback).
pub struct FileWeightIo {
    path: PathBuf,
    file: File,
}

impl FileWeightIo {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        tracing::debug!(path = %path.display(), "Opened File WeightIo");
        Ok(Self { path, file })
    }
}

impl WeightIo for FileWeightIo {
    fn path(&self) -> &Path {
        &self.path
    }

    fn backend(&self) -> IoBackend {
        IoBackend::File
    }

    fn file_size(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(0)
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(buf)
    }
}

#[cfg(target_os = "linux")]
mod iouring_weight {
    use super::*;
    use crate::buffer_pool::AlignedBuffer;
    use std::os::unix::fs::OpenOptionsExt;
    use std::os::unix::io::AsRawFd;
    use io_uring::{opcode, types, IoUring};

    /// `io_uring` weight reader with optional `O_DIRECT` (Phase F) and batched
    /// multi-read submission (Phase E).
    pub struct IoUringWeightIo {
        path: PathBuf,
        file: File,
        ring: IoUring,
        direct: bool,
        file_size: u64,
    }

    /// O_DIRECT alignment used for padded reads. 4096 is a multiple of both the
    /// 512-byte and 4096-byte logical block sizes, so it satisfies 4Kn devices.
    const DIRECT_ALIGN: u64 = 4096;

    impl IoUringWeightIo {
        pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
            let path = path.as_ref().to_path_buf();
            let want_direct = std::env::var_os("HAYAI_O_DIRECT").is_some();
            let (file, direct) = if want_direct {
                match std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)
                {
                    Ok(f) => {
                        tracing::info!("io_uring WeightIo: O_DIRECT enabled");
                        (f, true)
                    }
                    Err(e) => {
                        tracing::warn!("O_DIRECT open failed ({e}); falling back to buffered");
                        (File::open(&path)?, false)
                    }
                }
            } else {
                (File::open(&path)?, false)
            };
            let ring = IoUring::new(16).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            let file_size = file.metadata()?.len();
            // Streaming reads one layer at a time: ask the page cache for
            // sequential readahead (no-op/ignored under O_DIRECT).
            if !direct {
                let fd = file.as_raw_fd();
                let _ = unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_SEQUENTIAL) };
            }
            // Fork/prefetch opens many handles; keep INFO for catalog, DEBUG for each open.
            debug!(path = %path.display(), "Opened io_uring WeightIo");
            Ok(Self {
                path,
                file,
                ring,
                direct,
                file_size,
            })
        }

        fn read_exact_at(&mut self, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
            let fd = types::Fd(self.file.as_raw_fd());
            let mut filled = 0usize;
            while filled < buf.len() {
                let chunk = &mut buf[filled..];
                let len = chunk.len() as u32;
                let ptr = chunk.as_mut_ptr();
                let read_e = opcode::Read::new(fd, ptr, len)
                    .offset(offset)
                    .build()
                    .user_data(0x77);

                unsafe {
                    self.ring
                        .submission()
                        .push(&read_e)
                        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("sq push: {e}")))?;
                }
                self.ring.submit_and_wait(1)?;
                let cqe = self
                    .ring
                    .completion()
                    .next()
                    .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "missing cqe"))?;
                let n = cqe.result();
                if n < 0 {
                    return Err(io::Error::from_raw_os_error(-n));
                }
                if n == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "io_uring WeightIo EOF",
                    ));
                }
                let n = n as usize;
                offset += n as u64;
                filled += n;
            }
            Ok(())
        }

        /// O_DIRECT padded read: GGUF tensor offsets are only 32/256-aligned, but
        /// direct I/O requires offsets, lengths and buffer addresses aligned to the
        /// logical block size. Read the padded aligned range into a page-aligned
        /// scratch, then copy the requested slice. The padded range is clamped to
        /// the file size so the last tensor never triggers an out-of-range read.
        fn read_direct_padded(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            let start = offset & !(DIRECT_ALIGN - 1);
            let want_end = offset
                .checked_add(buf.len() as u64)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "offset overflow"))?;
            let end = want_end
                .saturating_add(DIRECT_ALIGN - 1)
                & !(DIRECT_ALIGN - 1);
            // A file whose size is not a multiple of the block size cannot be read
            // to its last byte with O_DIRECT; report it so `read_at` can fall back.
            if end > self.file_size {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "direct read would cross an unaligned EOF",
                ));
            }
            let n = (end - start) as usize;
            let mut scratch = AlignedBuffer::zeroed(n);
            self.read_exact_at(start, scratch.as_mut_slice())?;
            let skip = (offset - start) as usize;
            buf.copy_from_slice(&scratch.as_slice()[skip..skip + buf.len()]);
            Ok(())
        }
    }

    impl WeightIo for IoUringWeightIo {
        fn path(&self) -> &Path {
            &self.path
        }

        fn backend(&self) -> IoBackend {
            IoBackend::IoUring
        }

        fn file_size(&self) -> u64 {
            self.file_size
        }

        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            if self.direct {
                if offset
                    .checked_add(buf.len() as u64)
                    .map_or(true, |e| e > self.file_size)
                {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "io_uring read past EOF",
                    ));
                }
                let fully_aligned = offset % DIRECT_ALIGN == 0
                    && (buf.as_ptr() as usize as u64) % DIRECT_ALIGN == 0
                    && (buf.len() as u64) % DIRECT_ALIGN == 0;
                if fully_aligned {
                    return self.read_exact_at(offset, buf);
                }
                match self.read_direct_padded(offset, buf) {
                    Ok(()) => return Ok(()),
                    Err(e) => {
                        // O_DIRECT cannot serve this range (unaligned length, an
                        // unaligned tail, or a 4Kn device). Reopen buffered so the
                        // read is still correct — and say so instead of hiding it.
                        tracing::warn!(
                            "O_DIRECT read at {offset} failed ({e}); switching to buffered I/O"
                        );
                        self.file = File::open(&self.path)?;
                        self.direct = false;
                    }
                }
            }
            self.read_exact_at(offset, buf)
        }

        fn read_many_at(&mut self, dst: &mut [u8], requests: &[IoRange]) -> io::Result<()> {
            validate_ranges(dst.len(), requests)?;
            if self.direct {
                // Direct I/O pads each request; batch submission is per-request.
                for r in requests {
                    self.read_at(r.offset, &mut dst[r.start..r.end])?;
                }
                return Ok(());
            }
            let fd = types::Fd(self.file.as_raw_fd());
            let capacity = self.ring.submission().capacity().max(1);
            let mut done = 0usize;
            while done < requests.len() {
                let batch = (requests.len() - done).min(capacity);
                let mut pushed = 0usize;
                for (i, r) in requests[done..done + batch].iter().enumerate() {
                    // Temporary `&mut dst[..]` is dropped after `as_mut_ptr`; the raw
                    // pointers are used by the kernel after the borrows end. Ranges
                    // are validated disjoint above so the pointers never alias.
                    let ptr = dst[r.start..r.end].as_mut_ptr();
                    let len = (r.end - r.start) as u32;
                    let read_e = opcode::Read::new(fd, ptr, len)
                        .offset(r.offset)
                        .build()
                        .user_data((done + i) as u64);
                    let push_res = unsafe { self.ring.submission().push(&read_e) };
                    match push_res {
                        Ok(()) => pushed += 1,
                        Err(e) => {
                            // Drain anything already queued: leaving SQEs behind would
                            // let a later call submit them with dangling destination
                            // pointers.
                            let _ = self.ring.submit_and_wait(pushed);
                            while self.ring.completion().next().is_some() {}
                            return Err(io::Error::new(
                                io::ErrorKind::Other,
                                format!("sq push: {e}"),
                            ));
                        }
                    }
                }
                self.ring.submit_and_wait(pushed)?;
                // Drain every completion of the batch *before* interpreting any of
                // them, so an early error cannot leave CQEs behind for the next call.
                let mut results: Vec<(usize, i32)> = Vec::with_capacity(pushed);
                for _ in 0..pushed {
                    let cqe = self
                        .ring
                        .completion()
                        .next()
                        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "missing batch cqe"))?;
                    results.push((cqe.user_data() as usize, cqe.result()));
                }
                let mut seen = vec![false; batch];
                for (idx, n) in results {
                    let local = idx.checked_sub(done).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::Other, "unexpected cqe user_data")
                    })?;
                    if local >= batch || seen[local] {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "duplicate/out-of-range cqe user_data",
                        ));
                    }
                    seen[local] = true;
                    if n < 0 {
                        return Err(io::Error::from_raw_os_error(-n));
                    }
                    let want = requests[idx].end - requests[idx].start;
                    if n as usize != want {
                        return Err(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            "io_uring batch short read",
                        ));
                    }
                }
                done += batch;
            }
            Ok(())
        }
    }
}

#[cfg(target_os = "linux")]
use iouring_weight::IoUringWeightIo;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn file_weight_io_roundtrip() {
        let path = std::env::temp_dir().join("hayai_weight_io_test.bin");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&(0u8..64).collect::<Vec<_>>()).unwrap();
        }
        let mut io = open_weight_io(&path).unwrap();
        let mut buf = [0u8; 16];
        io.read_at(10, &mut buf).unwrap();
        assert_eq!(buf, (10u8..26).collect::<Vec<_>>().as_slice());
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_default_weight_io_is_io_uring() {
        let path = std::env::temp_dir().join("hayai_weight_io_uring.bin");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&(0u8..128).collect::<Vec<_>>()).unwrap();
        }
        // Ensure we exercise the production path (not the Tokio/File force-switch).
        std::env::remove_var("HAYAI_FORCE_TOKIO_IO");
        let mut io = open_weight_io(&path).unwrap();
        assert_eq!(
            io.backend(),
            IoBackend::IoUring,
            "Linux WeightIo must default to io_uring"
        );
        let mut buf = [0u8; 32];
        io.read_at(16, &mut buf).unwrap();
        assert_eq!(&buf[..], &(16u8..48).collect::<Vec<_>>()[..]);
        let _ = std::fs::remove_file(&path);
    }
}
