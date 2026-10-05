pub mod async_gemv;
pub mod attn;
pub mod compute;
pub mod context;
pub mod device;
pub mod hetero_scratch;
pub mod memory;
pub mod pool;

pub use async_gemv::PendingGemv;
pub use attn::{DeviceDeltanetState, DeviceKvCache};
pub use compute::GgmlWeightBind;
pub use context::{OpenClEngine, OpenClError};
pub use device::{discover_opencl_devices, DeviceKind, OpenClDeviceInfo};
/// Re-exported so callers can name the GEMV kernel type when selecting one.
pub use opencl3::kernel::Kernel;
pub use hetero_scratch::{StreamingScratch, WeightBind};
pub use memory::{
    select_transfer_path, DeviceLayerBuffer, OwnedSvmBuffer, PinnedHostBuffer, SvmLayerBuffer,
    SvmScratch, TransferPath,
};
pub use pool::OpenClDevicePool;
