use std::future::Future;
use std::io;
use std::path::Path;
use std::pin::Pin;
use std::time::Instant;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, BufReader, SeekFrom};
use tracing::{debug, info};

/// Statistics from a completed streaming session.
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub layers_streamed: usize,
    pub total_bytes: u64,
    pub elapsed_secs: f64,
    pub throughput_mbps: f64,
    /// Fraction of wall time spent blocked in I/O waits (0.0 = fully overlapped).
    pub io_stall_fraction: f64,
}

/// Which I/O backend is active for layer streaming.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoBackend {
    /// Async Tokio sequential layer reader (bench-io / Windows).
    Tokio,
    /// Blocking `File` seek+read (`WeightIo` fallback / Windows generate).
    File,
    #[cfg(target_os = "linux")]
    IoUring,
}

impl IoBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            IoBackend::Tokio => "tokio",
            IoBackend::File => "file",
            #[cfg(target_os = "linux")]
            IoBackend::IoUring => "io_uring",
        }
    }
}

/// Async layer reader abstraction.
///
/// Production target is Linux `io_uring`. The default implementation uses Tokio
/// so the same API works on the Windows development host.
pub trait LayerReader: Send {
    fn has_next(&self) -> bool;
    fn layer_size_bytes(&self) -> usize;
    fn total_layers(&self) -> usize;
    fn current_layer(&self) -> usize;
    fn backend(&self) -> IoBackend {
        IoBackend::Tokio
    }

    fn read_next_layer<'a>(
        &'a mut self,
        target_buffer: &'a mut [u8],
    ) -> Pin<Box<dyn Future<Output = io::Result<usize>> + Send + 'a>>;

    fn reset<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + 'a>>;

    fn stats(&self, total_elapsed: f64) -> StreamStats;
}

/// Open the preferred layer reader: `io_uring` on Linux unless `HAYAI_FORCE_TOKIO_IO=1`.
pub async fn open_layer_reader<P: AsRef<Path>>(
    path: P,
    layer_size_bytes: usize,
    total_layers: usize,
) -> io::Result<Box<dyn LayerReader>> {
    #[cfg(target_os = "linux")]
    {
        let force_tokio = std::env::var_os("HAYAI_FORCE_TOKIO_IO").is_some();
        if !force_tokio {
            match crate::iouring_reader::IoUringLayerPrefetcher::open(
                path.as_ref(),
                layer_size_bytes,
                total_layers,
            ) {
                Ok(p) => return Ok(Box::new(p)),
                Err(e) => {
                    tracing::warn!("io_uring open failed ({e}); falling back to Tokio I/O");
                }
            }
        }
    }
    let p = LayerPrefetcher::open(path, layer_size_bytes, total_layers).await?;
    Ok(Box::new(p))
}

/// Tokio-backed sequential layer prefetcher (dev-friendly; Linux prod prefers io_uring).
pub struct LayerPrefetcher {
    reader: BufReader<File>,
    layer_size_bytes: usize,
    total_layers: usize,
    current_layer: usize,
    total_io_wait_secs: f64,
}

impl LayerPrefetcher {
    pub async fn open<P: AsRef<Path>>(
        path: P,
        layer_size_bytes: usize,
        total_layers: usize,
    ) -> io::Result<Self> {
        let file = File::open(path.as_ref()).await?;
        let reader = BufReader::with_capacity(layer_size_bytes * 2, file);
        info!(
            "Opened Tokio streaming file {:?} — {} layers × {} KB each",
            path.as_ref(),
            total_layers,
            layer_size_bytes / 1024
        );
        Ok(Self {
            reader,
            layer_size_bytes,
            total_layers,
            current_layer: 0,
            total_io_wait_secs: 0.0,
        })
    }
}

impl LayerReader for LayerPrefetcher {
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
        IoBackend::Tokio
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
            self.reader
                .read_exact(&mut target_buffer[..self.layer_size_bytes])
                .await?;
            let io_secs = t0.elapsed().as_secs_f64();
            self.total_io_wait_secs += io_secs;
            self.current_layer += 1;
            debug!(
                "Layer {}/{} read in {:.2} ms ({} KB)",
                self.current_layer,
                self.total_layers,
                io_secs * 1000.0,
                self.layer_size_bytes / 1024
            );
            Ok(self.layer_size_bytes)
        })
    }

    fn reset<'a>(&'a mut self) -> Pin<Box<dyn Future<Output = io::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            self.reader.seek(SeekFrom::Start(0)).await?;
            self.current_layer = 0;
            self.total_io_wait_secs = 0.0;
            Ok(())
        })
    }

    fn stats(&self, total_elapsed: f64) -> StreamStats {
        let total_bytes = (self.current_layer * self.layer_size_bytes) as u64;
        let throughput_mbps =
            (total_bytes as f64 / (1024.0 * 1024.0)) / total_elapsed.max(1e-9);
        let io_stall_fraction = self.total_io_wait_secs / total_elapsed.max(1e-9);
        StreamStats {
            layers_streamed: self.current_layer,
            total_bytes,
            elapsed_secs: total_elapsed,
            throughput_mbps,
            io_stall_fraction,
        }
    }
}

/// Convenience inherent methods so existing call sites keep working without trait imports.
impl LayerPrefetcher {
    pub fn has_next(&self) -> bool {
        LayerReader::has_next(self)
    }

    pub async fn read_next_layer(&mut self, target_buffer: &mut [u8]) -> io::Result<usize> {
        LayerReader::read_next_layer(self, target_buffer).await
    }

    pub async fn reset(&mut self) -> io::Result<()> {
        LayerReader::reset(self).await
    }

    pub fn stats(&self, total_elapsed: f64) -> StreamStats {
        LayerReader::stats(self, total_elapsed)
    }
}
