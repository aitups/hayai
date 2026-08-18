pub mod buffer_pool;
pub mod reader;
pub mod weight_io;

#[cfg(target_os = "linux")]
pub mod iouring_reader;

pub use buffer_pool::{AlignedBuffer, PingPongBuffer};
pub use reader::{
    open_layer_reader, IoBackend, LayerPrefetcher, LayerReader, StreamStats,
};
pub use weight_io::{open_weight_io, FileWeightIo, WeightIo};

#[cfg(target_os = "linux")]
pub use iouring_reader::IoUringLayerPrefetcher;
