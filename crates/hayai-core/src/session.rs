//! Public session API for **token-by-token** generation (OpenAI-like server).
//!
//! A [`GenerationSession`] owns a fresh [`StreamingGenerator`] (independent KV
//! state) plus the session scratch, and borrows the shared [`EngineOrchestrator`]
//! (OpenCL pool) through an `Arc<Mutex<..>>` — requests for the same model are
//! serialized on the engine, while each request keeps its own context window.

use crate::adaptive_window::MemoryStrategy;
use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{StreamInferError, StreamingGenerator};
use hayai_model::{Grammar, GrammarState, GgufCatalog, Penalties, SamplerConfig, Tokenizer};
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
    grammar: Option<Grammar>,
    grammar_state: Option<GrammarState>,
    grammar_token_texts: Vec<Option<String>>,
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
        Ok(Self {
            gen,
            orch,
            scratch,
            last_logits: Vec::new(),
            prompt_tokens: 0,
            generated_ids: Vec::new(),
            finished: false,
            grammar: None,
            grammar_state: None,
            grammar_token_texts: Vec::new(),
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.gen.tokenizer
    }

    /// Set repetition/presence/frequency/logit-bias penalties for this session.
    pub fn set_penalties(&mut self, penalties: Penalties) {
        self.gen.penalties = penalties;
    }

    /// Enable grammar-constrained decoding for this session (GBNF).
    pub fn set_grammar(&mut self, grammar: Grammar) {
        self.grammar_token_texts = self.gen.tokenizer.token_texts();
        self.grammar_state = Some(grammar.initial_state());
        self.grammar = Some(grammar);
    }

    /// Mask the logits of tokens that would leave the grammar, and allow EOS only
    /// when the grammar may stop here.
    fn mask_logits_with_grammar(&self, logits: &mut [f32]) {
        let (Some(grammar), Some(state)) = (&self.grammar, &self.grammar_state) else {
            return;
        };
        for (id, text) in self.grammar_token_texts.iter().enumerate() {
            if id >= logits.len() {
                break;
            }
            if self.gen.tokenizer.is_stop(id as u32) {
                if !grammar.is_accepting(state) {
                    logits[id] = f32::NEG_INFINITY;
                }
                continue;
            }
            match text {
                Some(t) if !t.is_empty() => {
                    if grammar.advance_str(state, t).is_none() {
                        logits[id] = f32::NEG_INFINITY;
                    }
                }
                _ => logits[id] = f32::NEG_INFINITY,
            }
        }
    }

    fn advance_grammar(&mut self, token: u32) {
        let (Some(grammar), Some(state)) = (&self.grammar, self.grammar_state.take()) else {
            return;
        };
        let next = self
            .grammar_token_texts
            .get(token as usize)
            .and_then(|t| t.as_deref())
            .and_then(|t| grammar.advance_str(&state, t));
        self.grammar_state = Some(next.unwrap_or(state));
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
        if let Some(g) = &self.grammar {
            self.grammar_state = Some(g.initial_state());
        }
        Ok(ids.len())
    }

    /// Sample the next token and run one decode step, advancing the context.
    ///
    /// Returns `None` once generation is finished (EOS hit or already finished).
    pub fn next_token(&mut self) -> Result<Option<u32>, StreamInferError> {
        if self.finished {
            return Ok(None);
        }
        if self.grammar.is_some() {
            let mut logits = std::mem::take(&mut self.last_logits);
            self.mask_logits_with_grammar(&mut logits);
            self.last_logits = logits;
        }
        let sampler = self.gen.sampler;
        let next = hayai_model::sample_with(
            &self.last_logits,
            sampler,
            &mut self.gen.rng,
            &self.gen.penalties,
            &self.generated_ids,
        );
        self.advance_grammar(next);
        if self.gen.tokenizer.is_stop(next) {
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
