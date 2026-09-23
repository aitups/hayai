pub mod adaptive_window;
pub mod calibration;
pub mod deltanet;
pub mod exec_plan;
pub mod layer_cfg;
pub mod gemma_infer;
pub mod hrm_infer;
pub mod hybrid_infer;
pub mod infer;
pub mod mamba_infer;
pub mod metrics;
pub mod moe_infer;
pub mod mtp;
pub mod orchestrator;
pub mod pipeline;
pub mod prefill_wave;
pub mod session;
pub mod spec_decode;
pub mod stream_infer;

pub use adaptive_window::{compute_window_plan, MemoryStrategy, WindowPlan};
pub use calibration::{calibrate, DeviceProfile, HwProfile};
pub use exec_plan::{build_exec_plan, ExecPlan, LayerOpKind, UnknownLayerOp};
pub use infer::{spmm_adj, FfnOverride, GenerateStats, Generator, InferError, SparseAdj};
pub use metrics::{format_bytes, process_rss_bytes, HayaiOwnedMemory, StreamingMemoryBudget};
pub use orchestrator::{EngineOrchestrator, ExecutionMode, OrchestratorError};
pub use pipeline::{
    run_decode_benchmark, run_overlap_probe, HeterogeneousPipeline, OverlapProbeStats,
    PipelineStats, SyntheticFfnWeights,
};
// FFN disperso (DAG irregular, GGUF de `saor`): carga y SpMM CSR.
pub use hayai_model::{
    load_sparse_dag, sparse_dag_to_csr, spmm_csr_cpu, spmm_dense_masked, SparseDagBlock,
};
pub use session::GenerationSession;
pub use stream_infer::{load_config, MediaInput, StreamInferError, StreamingGenerator};
