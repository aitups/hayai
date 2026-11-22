//! Public session API for **token-by-token** generation (OpenAI-like server).
//!
//! A [`GenerationSession`] owns a fresh [`StreamingGenerator`] (independent KV
//! state) plus the session scratch, and borrows the shared [`EngineOrchestrator`]
//! (OpenCL pool) through an `Arc<Mutex<..>>` — requests for the same model are
//! serialized on the engine, while each request keeps its own context window.

use crate::adaptive_window::MemoryStrategy;
use crate::orchestrator::EngineOrchestrator;
use crate::stream_infer::{StreamInferError, StreamingGenerator};
use hayai_model::{Grammar, GrammarState, Penalties, SamplerConfig, Tokenizer};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Cross-request cache of a model's final projection / tied embedding, shared through the
/// server's `ModelHandle` so it is read from disk once per **model** instead of once per
/// request (audit D.2.3). Empty until first needed (lazy).
#[derive(Default)]
pub struct HeadCache {
    pub output: Option<Arc<hayai_model::QuantMatrix>>,
    pub embed: Option<Arc<hayai_model::QuantMatrix>>,
}

/// One generation request: prefill once, then drive decode token by token.
pub struct GenerationSession {
    gen: StreamingGenerator,
    orch: Arc<Mutex<EngineOrchestrator>>,
    scratch: Option<hayai_opencl::StreamingScratch>,
    /// Where to return the scratch on drop so the next request reuses it (server).
    scratch_slot: Option<Arc<Mutex<Option<hayai_opencl::StreamingScratch>>>>,
    last_logits: Vec<f32>,
    prompt_tokens: usize,
    generated_ids: Vec<u32>,
    finished: bool,
    grammar: Option<Grammar>,
    grammar_state: Option<GrammarState>,
    grammar_token_texts: Vec<Option<String>>,
}

impl Drop for GenerationSession {
    fn drop(&mut self) {
        // Return the scratch to the model's slot so the next request reuses it.
        if let (Some(slot), Some(sc)) = (&self.scratch_slot, self.scratch.take()) {
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(sc);
        }
    }
}

impl GenerationSession {
    /// Open the model, build the session scratch (resident / macro-chunk /
    /// minimal) and bind to the shared engine orchestrator.
    ///
    /// `orch` must already be initialized with this model's [`crate::orchestrator::ExecutionMode`].
    pub fn start(
        model: impl AsRef<Path>,
        tokenizer: Arc<Tokenizer>,
        head_cache: Arc<Mutex<HeadCache>>,
        header: Option<Arc<hayai_model::GgufHeader>>,
        scratch_slot: Option<Arc<Mutex<Option<hayai_opencl::StreamingScratch>>>>,
        sinks: usize,
        window: usize,
        memory_strategy: MemoryStrategy,
        sampler: SamplerConfig,
        seed: u64,
        orch: Arc<Mutex<EngineOrchestrator>>,
    ) -> Result<Self, StreamInferError> {
        let path = model.as_ref();
        // Reuse the model's shared tokenizer and (when present) its parsed GGUF header
        // (audit D.2.3) instead of re-reading/re-parsing the header per request.
        let mut gen = match header.as_deref() {
            Some(h) => {
                StreamingGenerator::open_with_header(path, tokenizer, sinks, window, sampler, seed, h)?
            }
            None => StreamingGenerator::open(path, tokenizer, sinks, window, sampler, seed)?,
        };
        gen.set_memory_strategy(memory_strategy);
        gen.set_head_cache(head_cache);
        // Reuse the model's scratch across requests (audit D.2.3): a resident scratch
        // already holds the weights, so its per-request preload is skipped.
        let reuse = scratch_slot
            .as_ref()
            .and_then(|s| s.lock().unwrap_or_else(|e| e.into_inner()).take());
        let mut guard = orch.lock().unwrap_or_else(|e| e.into_inner());
        let scratch = gen.prepare_session_reuse(&mut guard, reuse)?;
        drop(guard);
        Ok(Self {
            gen,
            orch,
            scratch: Some(scratch),
            scratch_slot,
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
        self.last_logits = self.gen.prefill(&mut guard, &ids, self.scratch.as_mut().expect("session scratch"))?;
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
        self.last_logits = self.gen.decode_step(&mut guard, next, self.scratch.as_mut().expect("session scratch"))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::ExecutionMode;
    use hayai_model::{GgufCatalog, SamplerConfig, Tokenizer};
    use std::path::PathBuf;

    fn acceptance_model() -> Option<PathBuf> {
        std::env::var("HAYAI_ACCEPTANCE_MODEL")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists())
    }

    #[test]
    fn resident_scratch_is_reused_across_sessions() {
        let Some(path) = acceptance_model() else {
            eprintln!("session reuse: SKIP (set HAYAI_ACCEPTANCE_MODEL)");
            return;
        };
        let cat = GgufCatalog::open(&path).unwrap();
        let tok = Arc::new(Tokenizer::from_catalog(&cat).unwrap());
        let header = Some(Arc::new(cat.header_snapshot()));
        let cfg = crate::load_config(&cat).unwrap();
        drop(cat);
        let orch = Arc::new(Mutex::new(EngineOrchestrator::new(
            ExecutionMode::CpuOnly,
            cfg,
        )));
        let head = Arc::new(Mutex::new(HeadCache::default()));
        let slot: Arc<Mutex<Option<hayai_opencl::StreamingScratch>>> = Arc::new(Mutex::new(None));
        let make = || {
            GenerationSession::start(
                &path,
                tok.clone(),
                head.clone(),
                header.clone(),
                Some(slot.clone()),
                4,
                512,
                MemoryStrategy::CapBytes(2_000_000_000),
                SamplerConfig::Greedy,
                0,
                orch.clone(),
            )
        };
        let run = |s: &mut GenerationSession| -> Vec<u32> {
            s.prefill("The capital of France is").unwrap();
            let mut ids = Vec::new();
            for _ in 0..8 {
                match s.next_token().unwrap() {
                    Some(t) => ids.push(t),
                    None => break,
                }
            }
            ids
        };
        let mut s1 = make().unwrap();
        let first = run(&mut s1);
        drop(s1);
        assert!(slot.lock().unwrap().is_some(), "scratch returned to the slot");
        assert!(!first.is_empty(), "expected some generated tokens");
        // Second session reuses the resident scratch (no re-preload) and is identical.
        let mut s2 = make().unwrap();
        let second = run(&mut s2);
        assert_eq!(first, second, "reused resident scratch changed the output");
    }
}

