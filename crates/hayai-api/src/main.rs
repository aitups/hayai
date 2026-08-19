use clap::Parser;
use hayai_api::{registry::ModelRegistry, router};
use hayai_core::{ExecutionMode, MemoryStrategy};
use std::path::Path;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "hayai-server")]
#[command(about = "Hayai — OpenAI-compatible LLM server (weight-streaming engine)")]
struct Args {
    /// GGUF model path(s) to serve (repeatable)
    #[arg(long)]
    model: Vec<String>,
    /// Directory scanned for *.gguf (default: models)
    #[arg(long, default_value = "models")]
    models_dir: String,
    /// HuggingFace repo id, e.g. bartowski/SmolLM2-135M-Instruct-GGUF
    /// (downloads the *Q4_K_M.gguf into --models-dir)
    #[arg(long)]
    hf: Vec<String>,
    /// Exact GGUF file inside the --hf repo (with --hf)
    #[arg(long)]
    hf_file: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value = "8080")]
    port: u16,
    /// auto | cpu | <OpenCL device name substring>
    #[arg(long, default_value = "auto")]
    device: String,
    /// auto | minimal | cap_mb (server default minimal avoids per-request resident preload)
    #[arg(long, default_value = "minimal")]
    memory_strategy: String,
    #[arg(long, default_value = "4")]
    sinks: usize,
    #[arg(long, default_value = "256")]
    window: usize,
    /// Chat template override: template string or @path/to/file
    #[arg(long)]
    chat_template: Option<String>,
    #[arg(long, default_value = "info")]
    log: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&args.log)),
        )
        .init();

    let mode = ExecutionMode::parse(&args.device);
    let strategy = MemoryStrategy::parse(&args.memory_strategy);
    let chat_override = args.chat_template.as_deref().map(|t| {
        if let Some(p) = t.strip_prefix('@') {
            std::fs::read_to_string(p).unwrap_or_else(|_| t.to_string())
        } else {
            t.to_string()
        }
    });

    let mut registry = ModelRegistry::new(mode, args.sinks, args.window, strategy, chat_override);

    let models_dir = Path::new(&args.models_dir);
    if models_dir.exists() {
        registry.scan_dir(models_dir);
    }
    for m in &args.model {
        registry.register(m);
    }
    for repo in &args.hf {
        let path = hayai_api::hf::download_model(repo, args.hf_file.as_deref(), models_dir)?;
        registry.register(path);
    }

    let n = registry.len();
    if n == 0 {
        println!(
            "No models registered. Serve with --model <file.gguf>, put GGUFs in '{}', or use --hf <repo>.",
            args.models_dir
        );
    } else {
        println!("Serving {} model(s): {:?}", n, registry.ids());
    }
    println!("Listening on http://{}:{}", args.host, args.port);

    let app = router(Arc::new(registry));
    let listener = tokio::net::TcpListener::bind((args.host.as_str(), args.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
