//! Model registry: auto-scan of `models/*.gguf` + explicit `--model` paths +
//! HuggingFace `--hf` downloads. Handles are built lazily (first request) and
//! cached; the OpenCL orchestrator is initialized once per model.

use crate::chat::ChatTemplate;
use crate::models::ApiError;
use hayai_core::{ExecutionMode, MemoryStrategy};
use hayai_model::{GgufCatalog, Tokenizer};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A loaded model: cached tokenizer + chat template + shared engine orchestrator.
pub struct ModelHandle {
    pub id: String,
    pub path: PathBuf,
    pub tokenizer: Arc<Tokenizer>,
    pub orch: Arc<Mutex<hayai_core::EngineOrchestrator>>,
    pub chat_template: ChatTemplate,
    pub sinks: usize,
    pub window: usize,
    pub memory_strategy: MemoryStrategy,
}

/// Server-wide registry (models discovered at startup; handles built lazily).
pub struct ModelRegistry {
    specs: HashMap<String, PathBuf>,
    handles: Mutex<HashMap<String, Arc<ModelHandle>>>,
    pub device: ExecutionMode,
    pub sinks: usize,
    pub window: usize,
    pub memory_strategy: MemoryStrategy,
    pub chat_template_override: Option<String>,
}

impl ModelRegistry {
    pub fn new(
        device: ExecutionMode,
        sinks: usize,
        window: usize,
        memory_strategy: MemoryStrategy,
        chat_template_override: Option<String>,
    ) -> Self {
        Self {
            specs: HashMap::new(),
            handles: Mutex::new(HashMap::new()),
            device,
            sinks,
            window,
            memory_strategy,
            chat_template_override,
        }
    }

    /// Register a GGUF path; the model id is the file stem (deduped).
    pub fn register(&mut self, path: impl AsRef<Path>) -> &mut Self {
        let path = path.as_ref().to_path_buf();
        if let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().into_owned()) {
            self.specs.entry(stem.clone()).or_insert(path);
        }
        self
    }

    /// Register every `*.gguf` under `dir` (non-recursive).
    pub fn scan_dir(&mut self, dir: impl AsRef<Path>) -> &mut Self {
        let dir = dir.as_ref();
        if let Ok(rd) = std::fs::read_dir(dir) {
            let mut found: Vec<PathBuf> = rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    p.extension()
                        .map(|e| e.to_string_lossy().eq_ignore_ascii_case("gguf"))
                        .unwrap_or(false)
                })
                .collect();
            found.sort();
            for p in found {
                self.register(p);
            }
        }
        self
    }

    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.specs.keys().cloned().collect();
        ids.sort();
        ids
    }

    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Get (lazily loading + caching) the handle for `id`.
    pub fn handle(&self, id: &str) -> Result<Arc<ModelHandle>, ApiError> {
        let path = self
            .specs
            .get(id)
            .cloned()
            .ok_or_else(|| ApiError::not_found(format!("model '{id}' not found")))?;
        if let Some(h) = self.handles.lock().unwrap().get(id) {
            return Ok(h.clone());
        }
        let handle = Arc::new(self.load_handle(id, path)?);
        self.handles.lock().unwrap().insert(id.to_string(), handle.clone());
        Ok(handle)
    }

    fn load_handle(&self, id: &str, path: PathBuf) -> Result<ModelHandle, ApiError> {
        let cat = GgufCatalog::open(&path).map_err(|e| ApiError::internal(e.to_string()))?;
        let tokenizer = Arc::new(Tokenizer::from_catalog(&cat).map_err(|e| ApiError::internal(e.to_string()))?);
        let config = hayai_core::load_config(&cat).map_err(|e| ApiError::internal(e.to_string()))?;
        drop(cat);
        let orch = hayai_core::EngineOrchestrator::new(self.device.clone(), config);
        let chat_template = ChatTemplate::from_model(&tokenizer, self.chat_template_override.as_deref());
        Ok(ModelHandle {
            id: id.to_string(),
            path,
            tokenizer,
            orch: Arc::new(Mutex::new(orch)),
            chat_template,
            sinks: self.sinks,
            window: self.window,
            memory_strategy: self.memory_strategy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_dedups_and_scans() {
        let mut reg = ModelRegistry::new(ExecutionMode::CpuOnly, 4, 128, MemoryStrategy::Minimal, None);
        reg.register("models/foo.gguf");
        reg.register("models/foo.gguf");
        assert_eq!(reg.ids(), vec!["foo".to_string()]);
        // scan_dir on a missing dir is a no-op
        reg.scan_dir("does_not_exist");
        assert_eq!(reg.len(), 1);
    }
}
