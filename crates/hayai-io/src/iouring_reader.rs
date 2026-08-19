//! Linux `io_uring` LayerReader (production I/O path).
//!
//! Compiled only on Linux. Windows / other hosts keep using [`super::LayerPrefetcher`] (Tokio).

use super::{IoBackend, LayerReader, StreamStats};
use std::fs::File;
use std::future::Future;
use std::io;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::pin::Pin;
use std::time::Instant;
use tracing::{debug, info};

use io_uring::{opcode, types, IoUring};

/// Sequential layer prefetcher backed by `io_uring` Read.
pub struct IoUringLayerPrefetcher {
    file: File,
    ring: IoUring,
    layer_size_bytes: usize,
    total_layers: usize,
    current_layer: usize,
    total_io_wait_secs: f64,
    file_offset: u64,
}

impl IoUringLayerPrefetcher {
    pub fn open<P: AsRef<Path>>(
        path: P,
        layer_size_bytes: usize,
        total_layers: usize,
    ) -> io::Result<Self> {
        let path = path.as_ref();
        // Phase F: optional O_DIRECT (bypass page cache); buffers are page-aligned
        // and layer offsets are sequential/aligned, so direct reads are legal.
        let want_direct = std::env::var_os("HAYAI_O_DIRECT").is_some();
        let file = if want_direct {
            match std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(path)
            {
                Ok(f) => {
                    info!("io_uring LayerReader: O_DIRECT enabled");
                    f
                }
                Err(e) => {
                    tracing::warn!("O_DIRECT layer reader open failed ({e}); buffered");
                    File::open(path)?
                }
            }
        } else {
            File::open(path)?
        };
        // Small ring; one in-flight read per layer is enough for the ping-pong pipeline.
        let ring = IoUring::new(8).map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        info!(
            "Opened io_uring streaming file {:?} — {} layers × {} KB",
            path,
            total_layers,
            layer_size_bytes / 1024
        );
        Ok(Self {
            file,
            ring,
            layer_size_bytes,
            total_layers,
            current_layer: 0,
            total_io_wait_secs: 0.0,
            file_offset: 0,
        })
    }

    fn read_exact_uring(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let fd = types::Fd(self.file.as_raw_fd());
        let mut filled = 0usize;
        while filled < buf.len() {
            let chunk = &mut buf[filled..];
            let len = chunk.len() as u32;
            let ptr = chunk.as_mut_ptr();
            // SAFETY: buffer lives until wait completes; single SQE outstanding.
            let read_e = opcode::Read::new(fd, ptr, len)
                .offset(self.file_offset)
                .build()
                .user_data(0x55);

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
                    "io_uring read EOF",
                ));
            }
            let n = n as usize;
            self.file_offset += n as u64;
            filled += n;
        }
        Ok(())
    }
}

impl LayerReader for IoUringLayerPrefetcher {
    fn has_next(&self) -> bool {
        self.current_layer < self.total_layers
    }

    fn layer_size_bytes(&self) -> usize {
        self.layer_size_bytes
    }

    fn total_layers(&self) -> usize {
        self.total_layers
    }

    fn current_layer(&self) -> usize {
        self.current_layer
    }

    fn backend(&self) -> IoBackend {
        IoBackend::IoUring
    }

    fn read_next_layer<'a>(
        &'a mut self,
        target_buffer: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>> {
        Box::pin(async move {
            assert!(
                target_buffer.len() >= self.layer_size_bytes,
                "Target buffer too small for layer"
            );
            let t0 = Instant::now();
            // io_uring submit/wait is blocking; run via spawn_blocking would need 'static.
            // For LayerReader contract we block the async worker briefly (same as sync I/O).
            self.read_exact_uring(&mut target_buffer[..self.layer_size_bytes])?;
            let io_secs = t0.elapsed().as_secs_f64();
            self.total_io_wait_secs += io_secs;
            self.current_layer += 1;
            debug!(
                "io_uring layer {}/{} in {:.2} ms",
                self.current_layer,
                self.total_layers,
                io_secs * 1000.0
            );
            Ok(self.layer_size_bytes)
        })
    }

    fn reset<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.file_offset = 0;
            self.current_layer = 0;
            self.total_io_wait_secs = 0.0;
            Ok(())
        })
    }

    fn stats(&self, total_elapsed: f64) -> StreamStats {
        let total_bytes = (self.current_layer * self.layer_size_bytes) as u64;
        let throughput_mbps =
            (total_bytes as f64 / (1024.0 * 1024.0)) / total_elapsed.max(1e-9);
        StreamStats {
            layers_streamed: self.current_layer,
            total_bytes,
            elapsed_secs: total_elapsed,
            throughput_mbps,
            io_stall_fraction: self.total_io_wait_secs / total_elapsed.max(1e-9),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::open_layer_reader;
    use std::io::Write;
    use tokio::runtime::Runtime;

    #[test]
    fn io_uring_layer_prefetcher_reads_layers() {
        let layer = 4096usize;
        let layers = 3usize;
        let path = std::env::temp_dir().join("hayai_iouring_layers.bin");
        {
            let mut f = File::create(&path).unwrap();
            for i in 0..(layer * layers) {
                f.write_all(&[(i % 251) as u8]).unwrap();
            }
        }
        std::env::remove_var("HAYAI_FORCE_TOKIO_IO");
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let mut reader = open_layer_reader(&path, layer, layers).await.unwrap();
            assert_eq!(reader.backend(), IoBackend::IoUring);
            let mut buf = vec![0u8; layer];
            for li in 0..layers {
                let n = reader.read_next_layer(&mut buf).await.unwrap();
                assert_eq!(n, layer);
                assert_eq!(buf[0], ((li * layer) % 251) as u8);
            }
            assert!(!reader.has_next());
        });
        let _ = std::fs::remove_file(&path);
    }
}
