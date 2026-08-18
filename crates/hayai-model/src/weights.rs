use crate::config::ModelConfig;
use crate::gguf::GgufFile;
use crate::gguf_types::GgufError;
use crate::quant::QuantMatrix;
use std::sync::Arc;
use tracing::info;

/// Llama-like weights as mmap views into a shared GGUF (zero-copy packed tensors).
pub struct LlamaWeights {
    pub gguf: Arc<GgufFile>,
    pub config: ModelConfig,
    pub tok_embd: QuantMatrix,
    pub output_norm: Vec<f32>,
    pub output: Option<QuantMatrix>,
    pub layers: Vec<LayerWeights>,
    /// Mapped weight bytes (tensor payloads; OS pages, not a dense FP32 copy).
    pub packed_nbytes: usize,
}

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: QuantMatrix,
    pub wk: QuantMatrix,
    pub wv: QuantMatrix,
    pub wo: QuantMatrix,
    pub ffn_norm: Vec<f32>,
    pub gate: QuantMatrix,
    pub up: QuantMatrix,
    pub down: QuantMatrix,
}

impl LlamaWeights {
    pub fn load(gguf: GgufFile) -> Result<Self, GgufError> {
        let gguf = Arc::new(gguf);
        Self::load_arc(gguf)
    }

    pub fn load_arc(gguf: Arc<GgufFile>) -> Result<Self, GgufError> {
        let arch = gguf
            .meta_str("general.architecture")
            .unwrap_or("llama")
            .to_string();
        let prefix = if arch == "llama" || arch == "qwen2" || arch.contains("smollm") {
            "llama"
        } else {
            arch.as_str()
        };

        let hidden = gguf
            .meta_u32(&format!("{prefix}.embedding_length"))
            .or_else(|| gguf.meta_u32("llama.embedding_length"))
            .ok_or_else(|| GgufError::MissingKey("embedding_length".into()))?
            as usize;
        let layers_n = gguf
            .meta_u32(&format!("{prefix}.block_count"))
            .or_else(|| gguf.meta_u32("llama.block_count"))
            .ok_or_else(|| GgufError::MissingKey("block_count".into()))?
            as usize;
        let intermediate = gguf
            .meta_u32(&format!("{prefix}.feed_forward_length"))
            .or_else(|| gguf.meta_u32("llama.feed_forward_length"))
            .unwrap_or((hidden * 8 / 3) as u32) as usize;
        let n_heads = gguf
            .meta_u32(&format!("{prefix}.attention.head_count"))
            .or_else(|| gguf.meta_u32("llama.attention.head_count"))
            .unwrap_or(8) as usize;
        let n_kv = gguf
            .meta_u32(&format!("{prefix}.attention.head_count_kv"))
            .or_else(|| gguf.meta_u32("llama.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        let ctx = gguf
            .meta_u32(&format!("{prefix}.context_length"))
            .or_else(|| gguf.meta_u32("llama.context_length"))
            .unwrap_or(2048) as usize;
        let rope = gguf
            .meta_f32(&format!("{prefix}.rope.freq_base"))
            .or_else(|| gguf.meta_f32("llama.rope.freq_base"))
            .unwrap_or(10000.0);
        let eps = gguf
            .meta_f32(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .or_else(|| gguf.meta_f32("llama.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);

        let tok_embd = QuantMatrix::from_gguf(gguf.clone(), "token_embd.weight")?;
        let vocab = tok_embd.nrows;

        let config = ModelConfig {
            name: gguf
                .meta_str("general.name")
                .unwrap_or("gguf-model")
                .to_string(),
            num_layers: layers_n,
            hidden_size: hidden,
            intermediate_size: intermediate,
            num_attention_heads: n_heads,
            num_key_value_heads: n_kv,
            vocab_size: vocab,
            max_position_embeddings: ctx,
            rope_theta: rope,
            rms_norm_eps: eps,
            architecture: prefix.to_string(),
            hrm: None,
        };

        let output_norm = gguf.dequant_f32("output_norm.weight")?;
        let output = match QuantMatrix::from_gguf(gguf.clone(), "output.weight") {
            Ok(w) => Some(w),
            Err(GgufError::MissingTensor(_)) => None,
            Err(e) => return Err(e),
        };

        let mut packed_nbytes = tok_embd.nbytes();
        if let Some(ref o) = output {
            packed_nbytes += o.nbytes();
        }

        let mut layers = Vec::with_capacity(layers_n);
        for i in 0..layers_n {
            info!("Mapping layer {i}/{layers_n}");
            let layer = LayerWeights {
                attn_norm: gguf.dequant_f32(&format!("blk.{i}.attn_norm.weight"))?,
                wq: QuantMatrix::from_gguf(gguf.clone(), &format!("blk.{i}.attn_q.weight"))?,
                wk: QuantMatrix::from_gguf(gguf.clone(), &format!("blk.{i}.attn_k.weight"))?,
                wv: QuantMatrix::from_gguf(gguf.clone(), &format!("blk.{i}.attn_v.weight"))?,
                wo: QuantMatrix::from_gguf(gguf.clone(), &format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: gguf.dequant_f32(&format!("blk.{i}.ffn_norm.weight"))?,
                gate: QuantMatrix::from_gguf(gguf.clone(), &format!("blk.{i}.ffn_gate.weight"))?,
                up: QuantMatrix::from_gguf(gguf.clone(), &format!("blk.{i}.ffn_up.weight"))?,
                down: QuantMatrix::from_gguf(gguf.clone(), &format!("blk.{i}.ffn_down.weight"))?,
            };
            packed_nbytes += layer.wq.nbytes()
                + layer.wk.nbytes()
                + layer.wv.nbytes()
                + layer.wo.nbytes()
                + layer.gate.nbytes()
                + layer.up.nbytes()
                + layer.down.nbytes();
            layers.push(layer);
        }

        let mapped = layers
            .iter()
            .all(|l| l.gate.is_mapped() && l.up.is_mapped() && l.down.is_mapped())
            && tok_embd.is_mapped();

        info!(
            "Loaded {} ({} layers, hidden={}, vocab={}, mapped_weights={:.1} MiB, zero_copy={})",
            config.name,
            layers_n,
            hidden,
            vocab,
            packed_nbytes as f64 / (1024.0 * 1024.0),
            mapped
        );

        Ok(Self {
            gguf,
            config,
            tok_embd,
            output_norm,
            output,
            layers,
            packed_nbytes,
        })
    }

    pub fn embed(&self, token: u32) -> Result<Vec<f32>, GgufError> {
        let h = self.config.hidden_size;
        let mut out = vec![0.0f32; h];
        self.tok_embd.embed_row(token, &mut out)?;
        Ok(out)
    }
}
