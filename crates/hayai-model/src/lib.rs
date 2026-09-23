#![feature(portable_simd)]

pub mod config;
pub mod cppn;
pub mod gguf;
pub mod gguf_stream;
pub mod gguf_types;
pub mod grammar;
pub mod iq2;
pub mod iq3;
pub mod iq4;
pub mod q2k;
pub mod q3k;
pub mod q4k;
pub mod q5;
pub mod q5k;
pub mod q6k;
pub mod quant;
pub mod sampling;
pub mod sparse_dag;
pub mod tokenizer;
pub mod vision;
pub mod weights;

pub use config::{HrmConfig, ModelConfig};
pub use cppn::{cppn_eval, layer_active_mask, layer_coord, sparse_layer_csr};
pub use gguf::{parse_header_bytes, tensor_nbytes, GgufFile, GgufHeader};
pub use gguf_stream::{GgufCatalog, LayerPackLayout, LayerWeightPack};
pub use gguf_types::{GgmlType, GgufError, MetadataValue, TensorInfo};
pub use grammar::{Grammar, GrammarError, GrammarState};
pub use quant::QuantMatrix;
pub use sampling::{sample, sample_with, Penalties, SamplerConfig};
pub use sparse_dag::{
    load_embedded_block, load_sparse_dag, sparse_dag_to_csr, spmm_csr_cpu, spmm_dense_masked,
    SparseDagBlock,
};
pub use tokenizer::Tokenizer;
pub use vision::{decode_audio_16k, load_image_rgb8, ClipEmbedder, MediaEmbeddings};
pub use weights::{CsrSparse, LayerWeights, LlamaWeights};
