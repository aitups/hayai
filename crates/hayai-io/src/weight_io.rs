//! Random-access weight I/O for GGUF streaming (PRD §3.1).
//!
//! Linux production: `io_uring` Read at absolute offsets.
//! Windows / fallback: blocking `File` seek+read (dev host).

use super::IoBackend;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
#[cfg(target_os = "linux")]
use tracing::{debug, info};

/// Deterministic weight reads into caller buffers (no mmap of payloads).
pub trait WeightIo: Send {
    fn path(&self) -> &Path;
    fn backend(&self) -> IoBackend;
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Batch reads into disjoint regions of `dst` (Phase E: `io_uring` submits all
    /// requests at once and waits once, instead of one syscall wait per tensor).
    /// Default implementation falls back to sequential [`Self::read_at`].
    fn read_many_at(&mut self, dst: &mut [u8], requests: &[IoRange]) -> io::Result<()> {
        for r in requests {
            self.read_at(r.offset, &mut dst[r.start..r.end])?;
        }
        Ok(())
    }
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
    }

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
            // Fork/prefetch opens many handles; keep INFO for catalog, DEBUG for each open.
            debug!(path = %path.display(), "Opened io_uring WeightIo");
            Ok(Self {
                path,
                file,
                ring,
                direct,
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

        /// O_DIRECT padded read: GGUF tensor offsets are 32/256-aligned, but the
        /// kernel requires 512-byte aligned offsets for direct I/O. Read the padded
        /// aligned range into a page-aligned scratch, then copy the requested slice.
        fn read_direct_padded(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            const SECTOR: u64 = 512;
            let start = offset & !(SECTOR - 1);
            let end = offset
                .saturating_add(buf.len() as u64)
                .saturating_add(SECTOR - 1)
                & !(SECTOR - 1);
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

        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            if self.direct {
                if offset % 512 == 0 && buf.as_ptr() as usize % 512 == 0 {
                    return self.read_exact_at(offset, buf);
                }
                return self.read_direct_padded(offset, buf);
            }
            self.read_exact_at(offset, buf)
        }

        fn read_many_at(&mut self, dst: &mut [u8], requests: &[IoRange]) -> io::Result<()> {
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
                for r in &requests[done..done + batch] {
                    // Temporary `&mut dst[..]` is dropped after `as_mut_ptr`; the raw
                    // pointers are used by the kernel after the borrows end. Ranges
                    // are disjoint (caller guarantee) so the pointers never alias.
                    let ptr = dst[r.start..r.end].as_mut_ptr();
                    let len = (r.end - r.start) as u32;
                    let read_e = opcode::Read::new(fd, ptr, len)
                        .offset(r.offset)
                        .build()
                        .user_data(0x78);
                    unsafe {
                        self.ring
                            .submission()
                            .push(&read_e)
                            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("sq push: {e}")))?;
                    }
                }
                // Submit the whole batch, then wait for all its completions once.
                self.ring.submit_and_wait(batch)?;
                for r in &requests[done..done + batch] {
                    let cqe = self
                        .ring
                        .completion()
                        .next()
                        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "missing batch cqe"))?;
                    let n = cqe.result();
                    if n < 0 {
                        return Err(io::Error::from_raw_os_error(-n));
                    }
                    if n as usize != r.end - r.start {
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
