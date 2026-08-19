pub mod buffer_pool;
pub mod io_worker;
pub mod reader;
pub mod weight_io;

#[cfg(target_os = "linux")]
pub mod iouring_reader;

pub use buffer_pool::{AlignedBuffer, PingPongBuffer};
pub use io_worker::IoWorker;
pub use reader::{
    open_layer_reader, IoBackend, LayerPrefetcher, LayerReader, StreamStats,
};
pub use weight_io::{open_weight_io, FileWeightIo, IoRange, WeightIo};

#[cfg(target_os = "linux")]
pub use iouring_reader::IoUringLayerPrefetcher;
