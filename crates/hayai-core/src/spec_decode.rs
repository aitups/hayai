//! MTP speculative decoding: the NextN draft head proposes `k` tokens, the main
//! model verifies them in a single batched forward, and the accepted prefix is
//! committed (rolling back KV + recurrent state on rejection).
//!
//! Greedy-only: sampling is deterministic, so speculative output is identical to
//! non-speculative greedy — the win is fewer main-model weight-streaming passes.

use crate::hybrid_infer::prefill_hybrid_all;
use crate::infer::GenerateStats;
use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{ModelKind, StreamInferError, StreamingGenerator};
use hayai_cpu::LayerKvCache;
use std::time::Instant;

#[derive(Clone)]
struct SpecSnapshot {
    kv: Vec<LayerKvCache>,
    deltanet: Option<Vec<Option<crate::deltanet::DeltaNetState>>>,
    position: usize,
}

fn argmax_id(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

impl StreamingGenerator {
    fn spec_snapshot(&self) -> SpecSnapshot {
        SpecSnapshot {
            kv: self.kv.clone(),
            deltanet: self.deltanet_states.clone(),
            position: self.position,
        }
    }

    fn spec_restore(&mut self, s: SpecSnapshot) {
        self.kv = s.kv;
        self.deltanet_states = s.deltanet;
        self.position = s.position;
    }

    /// Greedy speculative decoding with the MTP draft head (`n_draft >= 1`).
    pub fn generate_speculative(
        &mut self,
        orch: &mut EngineOrchestrator,
        prompt: &str,
        max_new_tokens: usize,
        n_draft: usize,
    ) -> Result<GenerateStats, StreamInferError> {
        if self.mtp_slot.is_none() {
            return Err(StreamInferError::Msg(
                "speculative decoding requires a NextN/MTP draft block".into(),
            ));
        }
        if self.model_kind() != ModelKind::Hybrid {
            return Err(StreamInferError::Msg(
                "MTP speculative decoding currently supports hybrid models only".into(),
            ));
        }
        if n_draft == 0 {
            return Err(StreamInferError::Msg("n_draft must be >= 1".into()));
        }
        let prompt_ids = self.tokenizer.encode(prompt, self.tokenizer.add_bos);
        if prompt_ids.is_empty() {
            return Err(StreamInferError::Msg("empty prompt tokenization".into()));
        }
        let prompt_len = prompt_ids.len();
        let stop_ids: Vec<u32> = self.tokenizer.eos_ids.clone();
        let hdim = self.config.hidden_size;
        let mut scratch = self.prepare_session(orch)?;
        let wall0 = Instant::now();

        // Prefill the prompt (per-token logits + post-norm hidden).
        let (all_logits, all_h) = prefill_hybrid_all(self, orch, &prompt_ids, &mut scratch)?;

        // Prime the MTP block's KV over the prompt: position i = token[i], h[i-1].
        for i in 0..prompt_len {
            let h = if i == 0 {
                vec![0.0f32; hdim]
            } else {
                all_h[i - 1].clone()
            };
            let _ = self.forward_mtp(orch, &h, prompt_ids[i], i)?;
        }

        let mut main_logits = all_logits.last().cloned().unwrap_or_default();
        let mut hidden = all_h.last().cloned().unwrap_or(vec![0.0f32; hdim]);
        let mut all_new: Vec<u32> = Vec::new();

        while all_new.len() < max_new_tokens {
            let mut hit_stop = false;
            let x1 = argmax_id(&main_logits);
            if stop_ids.contains(&x1) {
                break;
            }
            let hidden_before = hidden.clone();
            let pos0 = self.position;

            // Draft `n_draft` tokens by chaining the MTP head.
            let mut drafts: Vec<u32> = Vec::with_capacity(n_draft);
            let (mut dl, mut dh) = self.forward_mtp_full(orch, &hidden, x1, pos0)?;
            for j in 0..n_draft {
                let y = argmax_id(&dl);
                drafts.push(y);
                if stop_ids.contains(&y) || j + 1 == n_draft {
                    break;
                }
                let (dl2, dh2) = self.forward_mtp_full(orch, &dh, y, pos0 + 1 + j)?;
                dl = dl2;
                dh = dh2;
            }
            let k = drafts.len();

            // Verify the whole batch (x1 + drafts) in one main-model forward.
            let snap = self.spec_snapshot();
            let mut batch: Vec<u32> = Vec::with_capacity(k + 1);
            batch.push(x1);
            batch.extend_from_slice(&drafts);
            let (p, hb) = prefill_hybrid_all(self, orch, &batch, &mut scratch)?;

            let mut a = 0usize;
            while a < k && argmax_id(&p[a]) == drafts[a] {
                a += 1;
            }

            let committed: Vec<u32> = {
                let mut c = Vec::with_capacity(a + 1);
                c.push(x1);
                c.extend_from_slice(&drafts[..a]);
                c
            };

            if a == k {
                // All drafts accepted: the batched forward already committed their KV.
                for &t in &committed {
                    if stop_ids.contains(&t) {
                        hit_stop = true;
                        break;
                    }
                    all_new.push(t);
                    if all_new.len() >= max_new_tokens {
                        break;
                    }
                }
                // Catch the MTP KV up by one: it processed x1..y_{k-1} while drafting.
                if k >= 1 {
                    let _ = self.forward_mtp_full(orch, &hb[k - 1], drafts[k - 1], pos0 + k)?;
                }
                main_logits = p[k].clone();
                hidden = hb[k].clone();
            } else {
                // Rejected: roll back and commit only the accepted prefix.
                self.spec_restore(snap);
                let (p2, h2) = prefill_hybrid_all(self, orch, &committed, &mut scratch)?;
                // Re-prime the MTP KV for the committed prefix.
                let mut hp = hidden_before.clone();
                for (idx, &tok) in committed.iter().enumerate() {
                    let _ = self.forward_mtp_full(orch, &hp, tok, pos0 + idx)?;
                    hp = h2[idx].clone();
                }
                for &t in &committed {
                    if stop_ids.contains(&t) {
                        hit_stop = true;
                        break;
                    }
                    all_new.push(t);
                    if all_new.len() >= max_new_tokens {
                        break;
                    }
                }
                main_logits = p2.last().cloned().unwrap_or_default();
                hidden = h2.last().cloned().unwrap_or(vec![0.0f32; hdim]);
            }

            if hit_stop {
                break;
            }
        }

        self.wall_compute_secs = wall0.elapsed().as_secs_f64();
        Ok(GenerateStats {
            text: self.tokenizer.decode(&all_new),
            prompt_tokens: prompt_len,
            new_tokens: all_new.len(),
            total_positions: self.position,
        })
    }
}
