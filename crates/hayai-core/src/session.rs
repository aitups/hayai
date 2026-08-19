//! Public session API for **token-by-token** generation (OpenAI-like server).
//!
//! A [`GenerationSession`] owns a fresh [`StreamingGenerator`] (independent KV
//! state) plus the session scratch, and borrows the shared [`EngineOrchestrator`]
//! (OpenCL pool) through an `Arc<Mutex<..>>` — requests for the same model are
//! serialized on the engine, while each request keeps its own context window.

use crate::adaptive_window::MemoryStrategy;
use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{StreamInferError, StreamingGenerator};
use hayai_model::{GgufCatalog, SamplerConfig, Tokenizer};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// One generation request: prefill once, then drive decode token by token.
pub struct GenerationSession {
    gen: StreamingGenerator,
    orch: Arc<Mutex<EngineOrchestrator>>,
    scratch: hayai_opencl::StreamingScratch,
    last_logits: Vec<f32>,
    prompt_tokens: usize,
    generated_ids: Vec<u32>,
    finished: bool,
    eos: u32,
}

impl GenerationSession {
    /// Open the model, build the session scratch (resident / macro-chunk /
    /// minimal) and bind to the shared engine orchestrator.
    ///
    /// `orch` must already be initialized with this model's [`crate::orchestrator::ExecutionMode`].
    pub fn start(
        model: impl AsRef<Path>,
        sinks: usize,
        window: usize,
        memory_strategy: MemoryStrategy,
        sampler: SamplerConfig,
        seed: u64,
        orch: Arc<Mutex<EngineOrchestrator>>,
    ) -> Result<Self, StreamInferError> {
        let path = model.as_ref();
        let cat = GgufCatalog::open(path)?;
        let tokenizer = Tokenizer::from_catalog(&cat)?;
        drop(cat);
        let mut gen = StreamingGenerator::open(path, tokenizer, sinks, window, sampler, seed)?;
        gen.set_memory_strategy(memory_strategy);
        let mut guard = orch.lock().unwrap_or_else(|e| e.into_inner());
        let scratch = gen.prepare_session(&mut guard)?;
        drop(guard);
        let eos = gen.tokenizer.eos_id;
        Ok(Self {
            gen,
            orch,
            scratch,
            last_logits: Vec::new(),
            prompt_tokens: 0,
            generated_ids: Vec::new(),
            finished: false,
            eos,
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.gen.tokenizer
    }

    pub fn model_name(&self) -> &str {
        &self.gen.config.name
    }

    /// Tokenize + prefill `prompt` and return the number of prompt tokens.
    pub fn prefill(&mut self, prompt: &str) -> Result<usize, StreamInferError> {
        if self.finished {
            return Ok(0);
        }
        let ids = self.gen.tokenizer.encode(prompt, self.gen.tokenizer.add_bos);
        if ids.is_empty() {
            return Err(StreamInferError::Msg("empty prompt tokenization".into()));
        }
        let mut guard = self.orch.lock().unwrap_or_else(|e| e.into_inner());
        self.last_logits = self.gen.prefill(&mut guard, &ids, &mut self.scratch)?;
        self.prompt_tokens = ids.len();
        Ok(ids.len())
    }

    /// Sample the next token and run one decode step, advancing the context.
    ///
    /// Returns `None` once generation is finished (EOS hit or already finished).
    pub fn next_token(&mut self) -> Result<Option<u32>, StreamInferError> {
        if self.finished {
            return Ok(None);
        }
        let sampler = self.gen.sampler;
        let next = hayai_model::sample(&self.last_logits, sampler, &mut self.gen.rng);
        if next == self.eos {
            self.finished = true;
            return Ok(None);
        }
        self.generated_ids.push(next);
        let mut guard = self.orch.lock().unwrap_or_else(|e| e.into_inner());
        self.last_logits = self.gen.decode_step(&mut guard, next, &mut self.scratch)?;
        Ok(Some(next))
    }

    /// Mark the session finished externally (e.g. `max_tokens` reached).
    pub fn mark_finished(&mut self) {
        self.finished = true;
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    pub fn completion_tokens(&self) -> usize {
        self.generated_ids.len()
    }

    pub fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }

    /// Decode the accumulated generated ids.
    pub fn generated_text(&self) -> String {
        self.gen.tokenizer.decode(&self.generated_ids)
    }

    /// Decode arbitrary ids (deltas etc.).
    pub fn decode(&self, ids: &[u32]) -> String {
        self.gen.tokenizer.decode(ids)
    }
}
