//! NextN / MTP (multi-token prediction) draft head.
//!
//! Qwen3.5 appends a draft block after the main trunk (`blk.{last}` with
//! `nextn.eh_proj`/`enorm`/`hnorm`/`shared_head_norm` + a full-attention layer).
//! Given the previous position's normed hidden state and the just-sampled token,
//! it predicts the **next-next** token — the foundation for speculative decoding.
//!
//! The main trunk skips this block (`layer_cfg` marks it `NextN`); its KV cache is
//! still allocated as a real full-attention cache so [`Self::forward_mtp`] can use it.

use crate::hybrid_infer::{ffn_apply, full_attn_apply};
use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{StreamInferError, StreamingGenerator};
use hayai_cpu::rms_norm;
use hayai_model::{LayerWeightPack, QuantMatrix};

/// Cached MTP block weights (loaded once; avoids per-draft disk reads).
pub(crate) struct MtpCache {
    pub(crate) pack: LayerWeightPack,
    pub(crate) enorm: Vec<f32>,
    pub(crate) hnorm: Vec<f32>,
    pub(crate) shared_norm: Vec<f32>,
    pub(crate) eh: QuantMatrix,
}

impl StreamingGenerator {
    /// Index of the trailing NextN/MTP draft block, if present.
    pub fn mtp_layer(&self) -> Option<usize> {
        self.mtp_slot
    }

    /// Load (once) the MTP block weights into `mtp_cache`.
    fn ensure_mtp_cache(&mut self) -> Result<(), StreamInferError> {
        if self.mtp_cache.is_some() {
            return Ok(());
        }
        let layer = self
            .mtp_layer()
            .ok_or_else(|| StreamInferError::Msg("model has no NextN/MTP block".into()))?;
        let enorm = self
            .catalog
            .dequant_f32(&format!("blk.{layer}.nextn.enorm.weight"))?;
        let hnorm = self
            .catalog
            .dequant_f32(&format!("blk.{layer}.nextn.hnorm.weight"))?;
        let shared_norm = self
            .catalog
            .dequant_f32(&format!("blk.{layer}.nextn.shared_head_norm.weight"))?;
        let eh = self
            .catalog
            .load_quant_matrix(&format!("blk.{layer}.nextn.eh_proj.weight"))?;
        let pack = self.load_pack(layer)?;
        self.mtp_cache = Some(MtpCache {
            pack,
            enorm,
            hnorm,
            shared_norm,
            eh,
        });
        Ok(())
    }

    /// Post-final-norm hidden of the last processed token (`t_h_nextn`, after
    /// `output_norm`) — the input the MTP head consumes. Saved by every forward
    /// path; valid right after a prefill/decode step.
    pub fn last_hidden_normed(&self) -> Vec<f32> {
        if self.last_hidden_nextn.is_empty() {
            // Fallback (should not happen after a forward): norm the residual stream.
            let mut h = self.act().to_vec();
            rms_norm(&mut h, &self.output_norm, self.config.rms_norm_eps);
            return h;
        }
        self.last_hidden_nextn.clone()
    }

    /// Run the MTP/NextN draft head: `(hidden_prev, embed(next_token)) -> logits`
    /// for the token after `next_token`. `pos` is the draft head's own position.
    pub fn forward_mtp(
        &mut self,
        orch: &mut EngineOrchestrator,
        hidden_prev: &[f32],
        next_token: u32,
        pos: usize,
    ) -> Result<Vec<f32>, StreamInferError> {
        Ok(self.forward_mtp_full(orch, hidden_prev, next_token, pos)?.0)
    }

    /// Like [`Self::forward_mtp`] but also returns the draft block's own output
    /// hidden (`shared_head_norm`), which chains into the next MTP draft step.
    pub fn forward_mtp_full(
        &mut self,
        orch: &mut EngineOrchestrator,
        hidden_prev: &[f32],
        next_token: u32,
        pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), StreamInferError> {
        let layer = self
            .mtp_layer()
            .ok_or_else(|| StreamInferError::Msg("model has no NextN/MTP block".into()))?;
        let h = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        self.ensure_mtp_cache()?;
        let cache = self.mtp_cache.take().expect("mtp cache");

        // enorm(embed(next_token)) ∥ hnorm(hidden_prev) -> eh_proj -> hidden.
        let mut e = vec![0.0f32; h];
        let mut hh = hidden_prev.to_vec();
        let res = (|| -> Result<(), StreamInferError> {
            self.embed_row("token_embd.weight", next_token, h, &mut e)?;
            rms_norm(&mut e, &cache.enorm, eps);
            if hh.len() != h {
                return Err(StreamInferError::Msg(format!(
                    "MTP hidden_prev len {} != hidden {h}",
                    hh.len()
                )));
            }
            rms_norm(&mut hh, &cache.hnorm, eps);
            Ok(())
        })();
        if let Err(e) = res {
            self.mtp_cache = Some(cache);
            return Err(e);
        }
        let mut cat = e;
        cat.extend_from_slice(&hh);
        let mut x = vec![0.0f32; h];
        orch.execute_quant_gemv(&cache.eh, &cat, &mut x)?;

        // The draft block itself (full attention + FFN), on its own KV cache.
        full_attn_apply(self, orch, layer, pos, eps, &cache.pack, &mut x)?;
        ffn_apply(self, orch, None, layer, eps, &cache.pack, None, &mut x, None)?;
        rms_norm(&mut x, &cache.shared_norm, eps);
        self.mtp_cache = Some(cache);

        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        if self.has_output_weight {
            if let Some(ow) = &self.resident_output {
                orch.execute_quant_gemv(ow, &x, &mut logits)?;
            } else {
                let ow = self.catalog.load_quant_matrix("output.weight")?;
                orch.execute_quant_gemv(&ow, &x, &mut logits)?;
            }
        } else if let Some(emb) = &self.resident_embed {
            orch.execute_quant_gemv(emb, &x, &mut logits)?;
        } else {
            let emb = self.catalog.load_quant_matrix("token_embd.weight")?;
            orch.execute_quant_gemv(&emb, &x, &mut logits)?;
        }
        Ok((logits, x))
    }
}
