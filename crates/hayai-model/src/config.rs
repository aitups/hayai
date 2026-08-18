/// Optional HRM-Text recurrence parameters (`general.architecture = hrm_text`).
#[derive(Debug, Clone)]
pub struct HrmConfig {
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub layers_per_stack: usize,
    pub embedding_scale: f32,
}

impl HrmConfig {
    /// Unique KV cache slots = layers_per_stack × H_cycles × (L_cycles + 1).
    pub fn kv_slots(&self) -> usize {
        self.layers_per_stack * self.h_cycles * (self.l_cycles + 1)
    }

    /// Physical L-stack block index for layer `i` in the stack (0..layers_per_stack).
    /// GGUF convention (sapient / llama.cpp): L = blk.0..L-1, H = blk.L..2L-1.
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
