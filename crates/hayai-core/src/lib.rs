pub mod adaptive_window;
pub mod deltanet;
pub mod exec_plan;
pub mod layer_cfg;
pub mod gemma_infer;
pub mod hrm_infer;
pub mod hybrid_infer;
pub mod infer;
pub mod metrics;
pub mod orchestrator;
pub mod pipeline;
pub mod prefill_wave;
pub mod session;
pub mod stream_infer;

pub use adaptive_window::{compute_window_plan, MemoryStrategy, WindowPlan};
pub use exec_plan::{build_exec_plan, ExecPlan, LayerOpKind, UnknownLayerOp};
pub use infer::{GenerateStats, Generator, InferError};
pub use metrics::{format_bytes, process_rss_bytes, HayaiOwnedMemory, StreamingMemoryBudget};
pub use orchestrator::{EngineOrchestrator, ExecutionMode, OrchestratorError};
pub use pipeline::{
    run_decode_benchmark, run_overlap_probe, HeterogeneousPipeline, OverlapProbeStats,
    PipelineStats, SyntheticFfnWeights,
};
pub use session::GenerationSession;
pub use stream_infer::{load_config, StreamInferError, StreamingGenerator};
