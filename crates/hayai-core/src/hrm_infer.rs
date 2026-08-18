//! HRM-Text recurrent forward (H_cycles × L_cycles + gated attention).
//!
//! Physical GGUF layout: `blk.0..L-1` = L-stack, `blk.L..2L-1` = H-stack.
//! KV slots follow transformers cache layout:
//! `slot(h,l,layer) = (h*(L_cycles+1)+l)*layers_per_stack + layer`.

use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{
    ffn_begin_gate_up_scratch, ffn_finish_scratch, StreamInferError, StreamingGenerator,
};
use hayai_cpu::{attention_decode_step, rms_norm};
use hayai_opencl::StreamingScratch;
use std::time::Instant;

impl StreamingGenerator {
    /// One token through the full HRM recurrence; advances `position` by 1.
    pub(crate) fn forward_hrm_token(
        &mut self,
        orch: &mut EngineOrchestrator,
        scratch: &mut StreamingScratch,
        token: u32,
    ) -> Result<Vec<f32>, StreamInferError> {
        let hrm = self
            .config
            .hrm
            .clone()
            .ok_or_else(|| StreamInferError::Msg("forward_hrm_token without HrmConfig".into()))?;
        let h = self.config.hidden_size;
        let eps = self.config.rms_norm_eps;
        let pos = self.position;

        // z_H = embed * scale
        let mut z_h = vec![0.0f32; h];
        {
            let t0 = Instant::now();
            self.catalog
                .read_embed_row("token_embd.weight", token, h, &mut z_h)?;
            self.io_secs += t0.elapsed().as_secs_f64();
            self.io_bytes += (h * 2) as u64;
        }
        let scale = hrm.embedding_scale;
        for v in &mut z_h {
            *v *= scale;
        }

        // z_L = z_l_init
        let mut z_l = self
            .z_l_init
            .clone()
            .unwrap_or_else(|| vec![0.0f32; h]);
        if z_l.len() != h {
            return Err(StreamInferError::Msg(format!(
                "hrm.z_l_init len {} != hidden {h}",
                z_l.len()
            )));
        }

        for hc in 0..hrm.h_cycles {
            for lc in 0..hrm.l_cycles {
                // z_L = L_module(z_L + z_H)
                let mut x: Vec<f32> = z_l
                    .iter()
                    .zip(z_h.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                for layer in 0..hrm.layers_per_stack {
                    let blk = hrm.l_block(layer);
                    let kv_slot = hrm.kv_slot_l(hc, lc, layer);
                    self.hrm_run_block(orch, scratch, blk, kv_slot, pos, eps, &mut x)?;
                }
                // HrmTextStack.final_norm (parameterless RMSNorm).
                rms_norm(&mut x, &self.output_norm, eps);
                z_l = x;
            }
            // z_H = H_module(z_H + z_L)
            let mut x: Vec<f32> = z_h
                .iter()
                .zip(z_l.iter())
                .map(|(a, b)| a + b)
                .collect();
            for layer in 0..hrm.layers_per_stack {
                let blk = hrm.h_block(layer);
                let kv_slot = hrm.kv_slot_h(hc, layer);
                self.hrm_run_block(orch, scratch, blk, kv_slot, pos, eps, &mut x)?;
            }
            rms_norm(&mut x, &self.output_norm, eps);
            z_h = x;
        }

        // Logits from final z_H (no extra model.norm in HF CausalLM beyond stack norms).
        let xn = z_h;
        let vocab = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab];
        if self.has_output_weight {
            let t0 = Instant::now();
            let ow = self.catalog.load_quant_matrix("output.weight")?;
            self.io_secs += t0.elapsed().as_secs_f64();
            self.io_bytes += ow.nbytes() as u64;
            orch.execute_quant_gemv(&ow, &xn, &mut logits)?;
        } else {
            let t0 = Instant::now();
            let emb = self.catalog.load_quant_matrix("token_embd.weight")?;
            self.io_secs += t0.elapsed().as_secs_f64();
            self.io_bytes += emb.nbytes() as u64;
            orch.execute_quant_gemv(&emb, &xn, &mut logits)?;
        }
        self.position += 1;
        Ok(logits)
    }

    /// Stream one physical block (Attn+gate+FFN) into residual `x`.
    fn hrm_run_block(
        &mut self,
        orch: &mut EngineOrchestrator,
        scratch: &mut StreamingScratch,
        blk: usize,
        kv_slot: usize,
        pos: usize,
        eps: f32,
        x: &mut [f32],
    ) -> Result<(), StreamInferError> {
        let h = x.len();
        let slot = blk % 2;
        let (pack, layout) = self.stage_pack(orch, scratch, slot, blk)?;
        self.begin_ffn_dma(orch, scratch, slot, &layout)?;

        let t_attn = Instant::now();
        let mut xn = x.to_vec();
        rms_norm(&mut xn, &self.layer_norms[blk].attn_norm, eps);

        let q_dim = self.attn_cfg.hidden_size();
        let kv_dim = self.attn_cfg.kv_dim();
        let mut q = vec![0.0f32; q_dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        pack.wq.gemv(&xn, &mut q)?;
        pack.wk.gemv(&xn, &mut k)?;
        pack.wv.gemv(&xn, &mut v)?;

        let mut attn_out = vec![0.0f32; q_dim];
        attention_decode_step(
            &self.attn_cfg,
            &mut self.kv[kv_slot],
            &mut q,
            &mut k,
            &v,
            pos,
            &mut attn_out,
        );

        // Sigmoid gate before o_proj (Qwen3-Next / HRM style).
        if let Some(ref gate_w) = pack.attn_gate {
            let mut gate = vec![0.0f32; q_dim];
            gate_w.gemv(&xn, &mut gate)?;
            for i in 0..q_dim {
                let s = 1.0 / (1.0 + (-gate[i]).exp());
                attn_out[i] *= s;
            }
        }

        let mut attn_proj = vec![0.0f32; h];
        pack.wo.gemv(&attn_out, &mut attn_proj)?;
        for i in 0..h {
            x[i] += attn_proj[i];
        }
        self.attn_secs += t_attn.elapsed().as_secs_f64();

        let mut xn = x.to_vec();
        rms_norm(&mut xn, &self.layer_norms[blk].ffn_norm, eps);
        self.ws_gate.fill(0.0);
        self.ws_up.fill(0.0);
        self.ws_down.fill(0.0);
        self.finish_ffn_unmap(orch, scratch, slot)?;

        let t_ffn = Instant::now();
        let inflight = ffn_begin_gate_up_scratch(
            orch,
            &pack.gate,
            &pack.up,
            &xn,
            &mut self.ws_gate,
            &mut self.ws_up,
            &mut self.used_dgpu,
            &mut self.used_apu,
            Some(scratch),
            slot,
            Some(&layout),
        )?;
        ffn_finish_scratch(
            orch,
            inflight,
            &pack.down,
            &mut self.ws_gate,
            &mut self.ws_up,
            &mut self.ws_down,
            &mut self.used_dgpu,
            Some(scratch),
            slot,
            Some(&layout),
        )?;
        self.ffn_secs += t_ffn.elapsed().as_secs_f64();
        for i in 0..h {
            x[i] += self.ws_down[i];
        }
        Ok(())
    }
}
