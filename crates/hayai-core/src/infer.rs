//! End-to-end decode using GGUF mmap weights: Attn on CPU, FFN on OpenCL when available.

use hayai_cpu::{attention_decode_step, rms_norm, AttentionConfig, LayerKvCache};
use hayai_model::{sample, spmm_csr_cpu, GgufError, LlamaWeights, SamplerConfig, Tokenizer};
use tracing::debug;

use crate::orchestrator::{EngineOrchestrator, OrchestratorError};

#[derive(Debug, thiserror::Error)]
pub enum InferError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Orchestrator(#[from] OrchestratorError),
    #[error("{0}")]
    Msg(String),
}

pub struct Generator {
    pub weights: LlamaWeights,
    pub tokenizer: Tokenizer,
    pub attn_cfg: AttentionConfig,
    pub kv: Vec<LayerKvCache>,
    pub sampler: SamplerConfig,
    pub rng: u64,
    pub position: usize,
}

impl Generator {
    pub fn new(
        weights: LlamaWeights,
        tokenizer: Tokenizer,
        sink: usize,
        window: usize,
        sampler: SamplerConfig,
        seed: u64,
    ) -> Self {
        let attn_cfg = AttentionConfig::from_model(
            weights.config.hidden_size,
            weights.config.num_attention_heads,
            weights.config.num_key_value_heads,
            weights.config.rope_theta,
        );
        let kv = (0..weights.config.num_layers)
            .map(|_| {
                LayerKvCache::new(
                    attn_cfg.num_kv_heads,
                    attn_cfg.head_dim,
                    sink,
                    window,
                )
            })
            .collect();
        Self {
            weights,
            tokenizer,
            attn_cfg,
            kv,
            sampler,
            rng: seed,
            position: 0,
        }
    }

    /// Prefill / decode one token id → logits over vocab.
    /// Attention + projections on CPU; FFN GEMV via orchestrator (OpenCL preferred).
    pub fn forward(
        &mut self,
        orch: &mut EngineOrchestrator,
        token: u32,
    ) -> Result<Vec<f32>, InferError> {
        let h = self.weights.config.hidden_size;
        let mut x = self.weights.embed(token)?;
        let pos = self.position;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let mut xn = x.clone();
            rms_norm(&mut xn, &layer.attn_norm, self.weights.config.rms_norm_eps);

            let q_dim = self.attn_cfg.hidden_size();
            let kv_dim = self.attn_cfg.kv_dim();
            let mut q = vec![0.0f32; q_dim];
            let mut k = vec![0.0f32; kv_dim];
            let mut v = vec![0.0f32; kv_dim];
            // Attn projections stay on CPU (PRD: CPU owns attention path).
            layer.wq.gemv(&xn, &mut q)?;
            layer.wk.gemv(&xn, &mut k)?;
            layer.wv.gemv(&xn, &mut v)?;

            let mut attn_out_heads = vec![0.0f32; q_dim];
            attention_decode_step(
                &self.attn_cfg,
                &mut self.kv[layer_idx],
                &mut q,
                &mut k,
                &v,
                pos,
                &mut attn_out_heads,
            );

            let mut attn_proj = vec![0.0f32; h];
            layer.wo.gemv(&attn_out_heads, &mut attn_proj)?;
            for i in 0..h {
                x[i] += attn_proj[i];
            }

            let mut xn = x.clone();
            rms_norm(&mut xn, &layer.ffn_norm, self.weights.config.rms_norm_eps);
            let ff = self.weights.config.intermediate_size;
            let mut gate = vec![0.0f32; ff];
            let mut up = vec![0.0f32; ff];
            let mut down = vec![0.0f32; h];

            // FFN disperso (D16): si el tensor denso fue sustituido por el bloque
            // embebido, se ejecuta el CSR del profesor en las posiciones activas;
            // si no, el matvec cuantizado (OpenCL cuando está disponible).
            if let Some(c) = &layer.gate_csr {
                gate.copy_from_slice(&spmm_csr_cpu(
                    &xn,
                    &c.row_ptr,
                    &c.col_idx,
                    &c.vals,
                    c.d_in,
                    c.d_out,
                ));
            } else {
                orch.execute_quant_gemv(&layer.gate, &xn, &mut gate)?;
            }
            if let Some(c) = &layer.up_csr {
                up.copy_from_slice(&spmm_csr_cpu(
                    &xn,
                    &c.row_ptr,
                    &c.col_idx,
                    &c.vals,
                    c.d_in,
                    c.d_out,
                ));
            } else {
                orch.execute_quant_gemv(&layer.up, &xn, &mut up)?;
            }
            for i in 0..ff {
                let g = gate[i];
                gate[i] = (g / (1.0 + (-g).exp())) * up[i];
            }
            if let Some(c) = &layer.down_csr {
                down.copy_from_slice(&spmm_csr_cpu(
                    &gate,
                    &c.row_ptr,
                    &c.col_idx,
                    &c.vals,
                    c.d_in,
                    c.d_out,
                ));
            } else {
                orch.execute_quant_gemv(&layer.down, &gate, &mut down)?;
            }
            for i in 0..h {
                x[i] += down[i];
            }
        }

        let mut xn = x;
        rms_norm(
            &mut xn,
            &self.weights.output_norm,
            self.weights.config.rms_norm_eps,
        );

        let vocab = self.weights.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        if let Some(ref ow) = self.weights.output {
            orch.execute_quant_gemv(ow, &xn, &mut logits)?;
        } else {
            orch.execute_quant_gemv(&self.weights.tok_embd, &xn, &mut logits)?;
        }

        self.position += 1;
        Ok(logits)
    }

    pub fn generate(
        &mut self,
        orch: &mut EngineOrchestrator,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<GenerateStats, InferError> {
        let prompt_ids = self.tokenizer.encode(prompt, self.tokenizer.add_bos);
        if prompt_ids.is_empty() {
            return Err(InferError::Msg("empty prompt tokenization".into()));
        }
        debug!("prompt tokens: {:?}", prompt_ids);

        let prompt_len = prompt_ids.len();
        let mut all_ids = prompt_ids.clone();
        for &tid in &prompt_ids[..prompt_ids.len().saturating_sub(1)] {
            let _ = self.forward(orch, tid)?;
        }
        let mut last = *prompt_ids.last().unwrap();

        let mut new_tokens = 0usize;
        for _ in 0..max_new_tokens {
            let logits = self.forward(orch, last)?;
            let next = sample(&logits, self.sampler, &mut self.rng);
            all_ids.push(next);
            new_tokens += 1;
            if next == self.tokenizer.eos_id {
                break;
            }
            last = next;
        }

        let text = self.tokenizer.decode(&all_ids[prompt_len..]);
        Ok(GenerateStats {
            text,
            prompt_tokens: prompt_len,
            new_tokens,
            total_positions: self.position,
        })
    }
}

#[derive(Debug, Clone)]
pub struct GenerateStats {
    pub text: String,
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub total_positions: usize,
}
