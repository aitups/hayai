/// Optional HRM-Text recurrence parameters (`general.architecture = hrm_text`).
#[derive(Debug, Clone)]
pub struct HrmConfig {
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub layers_per_stack: usize,
    pub embedding_scale: f32,
    /// `hrm_text.prefix_lm`: informational (prefix-LM training objective).
    pub prefix_lm: bool,
}

impl HrmConfig {
    /// Unique KV cache slots = layers_per_stack × H_cycles × (L_cycles + 1).
    pub fn kv_slots(&self) -> usize {
        self.layers_per_stack * self.h_cycles * (self.l_cycles + 1)
    }

    /// Physical L-stack block index for layer `i` in the stack (0..layers_per_stack).
    /// The L/H weights are **shared** across recurrence cycles; `block_count` in the
    /// GGUF is the logical (unrolled) count `h_cycles*(l_cycles+1)*layers_per_stack`,
    /// while the file only stores `2 * layers_per_stack` physical blocks.
    /// GGUF convention: L = blk.0..L-1, H = blk.L..2L-1.
    pub fn l_block(&self, layer: usize) -> usize {
        layer
    }

    /// Physical H-stack block index.
    pub fn h_block(&self, layer: usize) -> usize {
        self.layers_per_stack + layer
    }

    /// KV slot for L-stack invocation at `(h_cycle, l_cycle, layer)`.
    pub fn kv_slot_l(&self, h_cycle: usize, l_cycle: usize, layer: usize) -> usize {
        (h_cycle * (self.l_cycles + 1) + l_cycle) * self.layers_per_stack + layer
    }

    /// KV slot for trailing H-stack invocation at `(h_cycle, layer)`.
    pub fn kv_slot_h(&self, h_cycle: usize, layer: usize) -> usize {
        (h_cycle * (self.l_cycles + 1) + self.l_cycles) * self.layers_per_stack + layer
    }
}

/// Model Architecture Config (tailored for SmolLM-135M-Instruct and Llama-family LLMs).
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub architecture: String,
    pub hrm: Option<HrmConfig>,
}

impl ModelConfig {
    /// Returns default SmolLM-135M-Instruct parameters.
    pub fn smollm_135m() -> Self {
        Self {
            name: "HuggingFaceTB/SmolLM-135M-Instruct".into(),
            num_layers: 30,
            hidden_size: 576,
            intermediate_size: 1536,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            vocab_size: 49152,
            max_position_embeddings: 2048,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            architecture: "llama".into(),
            hrm: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hrm_text_1b() -> HrmConfig {
        HrmConfig {
            h_cycles: 2,
            l_cycles: 3,
            layers_per_stack: 16,
            embedding_scale: 39.19,
            prefix_lm: true,
        }
    }

    #[test]
    fn hrm_kv_slots_and_physical_blocks() {
        let h = hrm_text_1b();
        // Logical schedule = H_cycles * (L_cycles+1) * layers_per_stack.
        assert_eq!(h.kv_slots(), 128);
        // Physical blocks are shared across cycles: 2 stacks of 16.
        assert_eq!(h.l_block(0), 0);
        assert_eq!(h.l_block(15), 15);
        assert_eq!(h.h_block(0), 16);
        assert_eq!(h.h_block(15), 31);
    }

    #[test]
    fn hrm_kv_slots_are_unique_and_in_range() {
        let h = hrm_text_1b();
        let mut seen = std::collections::HashSet::new();
        for hc in 0..h.h_cycles {
            for lc in 0..h.l_cycles {
                for layer in 0..h.layers_per_stack {
                    let s = h.kv_slot_l(hc, lc, layer);
                    assert!(s < h.kv_slots(), "slot {s} out of range");
                    assert!(seen.insert(s), "duplicate L slot {s}");
                }
            }
            for layer in 0..h.layers_per_stack {
                let s = h.kv_slot_h(hc, layer);
                assert!(s < h.kv_slots(), "slot {s} out of range");
                assert!(seen.insert(s), "duplicate H slot {s}");
            }
        }
        assert_eq!(seen.len(), h.kv_slots());
    }
}
