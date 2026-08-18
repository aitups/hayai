//! Random-access weight I/O for GGUF streaming (PRD §3.1).
//!
//! Linux production: `io_uring` Read at absolute offsets.
//! Windows / fallback: blocking `File` seek+read (dev host).

use super::IoBackend;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// Deterministic weight reads into caller buffers (no mmap of payloads).
pub trait WeightIo: Send {
    fn path(&self) -> &Path;
    fn backend(&self) -> IoBackend;
    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
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
    use std::os::unix::io::AsRawFd;
    use io_uring::{opcode, types, IoUring};

    pub struct IoUringWeightIo {
        path: PathBuf,
        file: File,
        ring: IoUring,
    }

    impl IoUringWeightIo {
        pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
            let path = path.as_ref().to_path_buf();
            let file = File::open(&path)?;
            let ring = IoUring::new(8).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            // Fork/prefetch opens many handles; keep INFO for catalog, DEBUG for each open.
            debug!(path = %path.display(), "Opened io_uring WeightIo");
            Ok(Self { path, file, ring })
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
    }

    impl WeightIo for IoUringWeightIo {
        fn path(&self) -> &Path {
            &self.path
        }

        fn backend(&self) -> IoBackend {
            IoBackend::IoUring
        }

        fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            self.read_exact_at(offset, buf)
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
