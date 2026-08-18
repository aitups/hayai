use clap::{Parser, Subcommand};
use hayai_core::{
    build_exec_plan, format_bytes, process_rss_bytes, run_decode_benchmark, run_overlap_probe,
    EngineOrchestrator, ExecutionMode, Generator, MemoryStrategy, StreamingGenerator,
    StreamingMemoryBudget,
};
use hayai_cpu::{cpu_lut_matmul_q4, fp32_matmul, max_abs_diff, unpack_q4_to_fp32};
use hayai_io::{open_layer_reader, PingPongBuffer};
use hayai_model::{GgufFile, LlamaWeights, ModelConfig, SamplerConfig, Tokenizer};
use hayai_opencl::{
    discover_opencl_devices, select_transfer_path, OpenClEngine, StreamingScratch, TransferPath,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tokio::fs;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "hayai")]
#[command(about = "Hayai: Low-Memory LLM Weight-Streaming Engine (OpenCL 3.0 + Rust)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect available hardware (OpenCL APUs, GPUs, CPU SIMD)
    Info,

    /// Run MatMul micro-benchmark (OpenCL when available, else CPU)
    Bench {
        #[arg(long, default_value = "auto")]
        device: String,
    },

    /// Validate Q4 LUT MatMul: FP32 reference vs CPU vs OpenCL
    Validate {
        #[arg(long, default_value = "auto")]
        device: String,
    },

    /// Run the streaming I/O pipeline benchmark (Disk → Ping-Pong → compute/upload)
    BenchIo {
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value = "3")]
        passes: usize,
    },

    /// Phase-3 heterogeneous pipeline: CPU Attention (KV INT8) ∥ GPU FFN
    BenchHetero {
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value = "4")]
        tokens: usize,
        #[arg(long, default_value = "4")]
        sinks: usize,
        #[arg(long, default_value = "64")]
        window: usize,
        #[arg(long, default_value = "4")]
        layers: usize,
    },

    /// Download a SmolLM-135M Instruct GGUF into models/
    FetchModel {
        #[arg(long, default_value = "models")]
        dir: PathBuf,
        /// Override download URL
        #[arg(long)]
        url: Option<String>,
    },

    /// Generate text from a GGUF model (Phase 4/5 E2E)
    Generate {
        /// Path to .gguf file
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value = "32")]
        max_tokens: usize,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, default_value = "4")]
        sinks: usize,
        #[arg(long, default_value = "256")]
        window: usize,
        /// greedy | temperature | top_p
        #[arg(long, default_value = "greedy")]
        sample: String,
        #[arg(long, default_value = "0.8")]
        temperature: f32,
        #[arg(long, default_value = "0.9")]
        top_p: f32,
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Disable ChatML wrap (send prompt as-is)
        #[arg(long, default_value_t = false)]
        raw: bool,
        /// DEV ONLY: load weights via full-file mmap (violates PRD streaming design)
        #[arg(long, default_value_t = false)]
        dev_mmap: bool,
        /// Memory window strategy: auto | minimal | cap_mb (e.g. 2048 or 2048mb)
        #[arg(long, default_value = "auto")]
        memory_strategy: String,
    },

    /// Benchmark packed-Q generate: tok/s + packed weight MiB
    BenchGenerate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "Say hi in one word.")]
        prompt: String,
        #[arg(long, default_value = "16")]
        max_tokens: usize,
        #[arg(long, default_value = "cpu")]
        device: String,
        #[arg(long, default_value = "4")]
        sinks: usize,
        #[arg(long, default_value = "128")]
        window: usize,
        /// Disable ChatML wrap (send prompt as-is; better for fair llama.cpp compare)
        #[arg(long, default_value_t = false)]
        raw: bool,
        /// Memory window strategy: auto | minimal | cap_mb (e.g. 2048 or 2048mb)
        #[arg(long, default_value = "auto")]
        memory_strategy: String,
    },

    /// Parse GGUF metadata → streaming ExecPlan (ops registry + HW strategy)
    Plan {
        #[arg(long)]
        model: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Info => cmd_info(),
        Commands::Bench { device } => cmd_bench(device)?,
        Commands::Validate { device } => cmd_validate(device)?,
        Commands::BenchIo { device, passes } => cmd_bench_io(device, passes).await?,
        Commands::BenchHetero {
            device,
            tokens,
            sinks,
            window,
            layers,
        } => cmd_bench_hetero(device, tokens, sinks, window, layers)?,
        Commands::FetchModel { dir, url } => cmd_fetch_model(dir, url)?,
        Commands::Generate {
            model,
            prompt,
            max_tokens,
            device,
            sinks,
            window,
            sample,
            temperature,
            top_p,
            seed,
            raw,
            dev_mmap,
            memory_strategy,
        } => cmd_generate(
            model,
            prompt,
            max_tokens,
            device,
            sinks,
            window,
            sample,
            temperature,
            top_p,
            seed,
            !raw,
            dev_mmap,
            memory_strategy,
        )?,
        Commands::BenchGenerate {
            model,
            prompt,
            max_tokens,
            device,
            sinks,
            window,
            raw,
            memory_strategy,
        } => cmd_bench_generate(
            model,
            prompt,
            max_tokens,
            device,
            sinks,
            window,
            !raw,
            memory_strategy,
        )?,
        Commands::Plan { model } => cmd_plan(model)?,
    }

    Ok(())
}

fn cmd_plan(model: PathBuf) -> anyhow::Result<()> {
    let cat = hayai_model::GgufCatalog::open(&model)?;
    let devices = discover_opencl_devices();
    let n_gpu = devices
        .iter()
        .filter(|d| {
            !matches!(
                d.device_kind,
                hayai_opencl::DeviceKind::CpuOpenCl
            )
        })
        .count();
    let svm = devices.iter().any(|d| d.supports_svm);
    match build_exec_plan(&cat, n_gpu, svm) {
        Ok(plan) => {
            println!("============================================================");
            println!("               HAYAI EXEC PLAN");
            println!("============================================================");
            println!("model: {}", model.display());
            print!("{}", plan.format_report());
            println!("status: OK (all tensors mapped to known LayerOpKind)");
        }
        Err(u) => {
            println!("============================================================");
            println!("               HAYAI EXEC PLAN — BLOCKED");
            println!("============================================================");
            println!("model: {}", model.display());
            println!("unknown_op tensor: {}", u.tensor_name);
            println!("hint: {}", u.hint);
            println!("status: arch-blocked (unknown layer op only)");
            anyhow::bail!("unknown layer op: {}", u.tensor_name);
        }
    }
    Ok(())
}

fn cmd_info() {
    println!("============================================================");
    println!("               HAYAI HARDWARE DISCOVERY");
    println!("============================================================");
    println!("Product target: Linux (multi-arch). Windows = dev host only.");
    let devices = discover_opencl_devices();
    if devices.is_empty() {
        println!("No OpenCL devices found. Hayai will run in CPU-Only mode.");
    } else {
        for (idx, dev) in devices.iter().enumerate() {
            let path = select_transfer_path(dev);
            println!("[Device #{}]", idx + 1);
            println!("  Name:    {}", dev.device_name);
            println!("  Platform:{}", dev.platform_name);
            println!("  Vendor:  {}", dev.vendor);
            println!("  Kind:    {:?}", dev.device_kind);
            println!("  SVM:     {}", dev.supports_svm);
            println!("  Path:    {:?}", path);
            println!("  Compute Units:   {}", dev.max_compute_units);
            println!(
                "  VRAM/RAM:        {} MB",
                dev.global_mem_size / (1024 * 1024)
            );
            println!("------------------------------------------------------------");
        }
    }
    println!("CPU backend: Rayon parallel Q4 LUT MatMul (+ FP32 reference)");
    println!("Test Model:  SmolLM-135M-Instruct");
    println!("============================================================");
}

fn cmd_bench(device: String) -> anyhow::Result<()> {
    let mode = parse_device_mode(&device);
    let config = ModelConfig::smollm_135m();
    let mut orchestrator = EngineOrchestrator::new(mode, config.clone());

    let m = config.intermediate_size;
    let n = config.hidden_size;
    let packed_cols = n / 2;

    let weights_q4 = vec![0xABu8; m * packed_cols];
    let lut: [f32; 16] = [
        -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ];
    let input = vec![1.0f32; n];
    let mut output = vec![0.0f32; m];

    for _ in 0..5 {
        orchestrator.execute_lut_matmul(m, n, &weights_q4, &lut, &input, &mut output)?;
    }

    let iterations = if orchestrator.using_opencl() { 100 } else { 500 };
    let start = Instant::now();
    for _ in 0..iterations {
        orchestrator.execute_lut_matmul(m, n, &weights_q4, &lut, &input, &mut output)?;
    }
    let elapsed = start.elapsed();
    let total_ops = 2usize * m * n * iterations;
    let gflops = (total_ops as f64 / 1e9) / elapsed.as_secs_f64();

    println!("─────────────────────────────────────────────────────");
    println!("  MatMul Benchmark: SmolLM-135M FFN Layer [{m}×{n}] Q4");
    println!("  Mode:       {:?}", orchestrator.mode);
    println!(
        "  Backend:    {}",
        if orchestrator.using_opencl() {
            "OpenCL"
        } else {
            "CPU"
        }
    );
    println!("  Iters:      {iterations}");
    println!("  Duration:   {:.3}s", elapsed.as_secs_f64());
    println!("  Throughput: {gflops:.2} GFLOPS");
    println!("─────────────────────────────────────────────────────");
    Ok(())
}

fn cmd_validate(device: String) -> anyhow::Result<()> {
    let mode = parse_device_mode(&device);
    let config = ModelConfig::smollm_135m();
    let mut orch = EngineOrchestrator::new(mode, config);

    let m = 128;
    let n = 256;
    let lut: [f32; 16] = [
        -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ];
    let weights_q4: Vec<u8> = (0..(m * n / 2))
        .map(|i| ((i * 31) % 256) as u8)
        .collect();
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 0.25).collect();

    let dense = unpack_q4_to_fp32(m, n, &weights_q4, &lut);
    let mut fp32_out = vec![0.0f32; m];
    let mut cpu_out = vec![0.0f32; m];
    let mut orch_out = vec![0.0f32; m];

    fp32_matmul(m, n, &dense, &input, &mut fp32_out);
    cpu_lut_matmul_q4(m, n, &weights_q4, &lut, &input, &mut cpu_out);
    orch.execute_lut_matmul(m, n, &weights_q4, &lut, &input, &mut orch_out)?;

    let cpu_err = max_abs_diff(&fp32_out, &cpu_out);
    let orch_err = max_abs_diff(&fp32_out, &orch_out);

    println!("─────────────────────────────────────────────────────");
    println!("  Precision validation [{m}×{n}] Q4 LUT MatMul");
    println!("  Mode:            {:?}", orch.mode);
    println!("  CPU vs FP32:     max_abs_diff = {cpu_err:.6e}");
    println!(
        "  Orchestrator vs FP32: max_abs_diff = {orch_err:.6e} ({})",
        if orch.using_opencl() {
            "OpenCL"
        } else {
            "CPU"
        }
    );
    println!("─────────────────────────────────────────────────────");

    anyhow::ensure!(cpu_err < 1e-4, "CPU Q4 diverges from FP32: {cpu_err}");
    anyhow::ensure!(
        orch_err < 1e-4,
        "Orchestrator diverges from FP32: {orch_err}"
    );
    println!("  OK");
    Ok(())
}

async fn cmd_bench_io(device: String, passes: usize) -> anyhow::Result<()> {
    let config = ModelConfig::smollm_135m();

    let gate_bytes = (config.intermediate_size * config.hidden_size) / 2;
    let down_bytes = (config.hidden_size * config.intermediate_size) / 2;
    let ffn_layer_bytes = gate_bytes + gate_bytes + down_bytes;
    let total_layers = config.num_layers;
    let model_total_bytes = (ffn_layer_bytes * total_layers) as u64;

    let tile_m = 128usize.min(config.intermediate_size);
    let tile_n = config.hidden_size;
    let tile_packed = tile_m * tile_n / 2;
    let lut: [f32; 16] = [
        -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    ];
    let input = vec![1.0f32; tile_n];
    let mut output = vec![0.0f32; tile_m];

    println!("─────────────────────────────────────────────────────────────");
    println!("  HAYAI Streaming I/O Benchmark");
    println!("  Model:       {}", config.name);
    println!("  Layers:      {total_layers}");
    println!(
        "  FFN/layer:   {} KB  (Q4 packed: gate+up+down)",
        ffn_layer_bytes / 1024
    );
    println!(
        "  Total model: {:.2} MB",
        model_total_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("─────────────────────────────────────────────────────────────");

    let tmp_path = std::env::temp_dir().join("hayai_bench_smollm.bin");
    println!("  Generating synthetic model file → {tmp_path:?}");
    let t_gen = Instant::now();
    {
        let synthetic_data: Vec<u8> = (0..model_total_bytes).map(|i| (i % 256) as u8).collect();
        fs::write(&tmp_path, &synthetic_data).await?;
    }
    println!("  Generated in {:.2}s", t_gen.elapsed().as_secs_f64());

    let use_gpu = device != "cpu";
    let cl_engine = if use_gpu {
        match OpenClEngine::try_init_any() {
            Ok(eng) => {
                let path = select_transfer_path(&eng.device_info);
                println!(
                    "  Upload target: {} ({:?})",
                    eng.device_info.device_name, path
                );
                Some(eng)
            }
            Err(e) => {
                println!("  No OpenCL device ({e}) → HostRam streaming + CPU compute");
                None
            }
        }
    } else {
        println!("  Upload target: CPU-Only (HostRam ping-pong)");
        None
    };

    let selected = cl_engine
        .as_ref()
        .map(|e| select_transfer_path(&e.device_info))
        .unwrap_or(TransferPath::HostRam);

    // Pinned/Host long-lived scratch; SVM exercised in a scoped block when available.
    let (transfer_path, mut scratch) =
        StreamingScratch::allocate(cl_engine.as_ref(), ffn_layer_bytes)?;
    let mut svm_scratch = if matches!(selected, TransferPath::SvmZeroCopy) {
        cl_engine
            .as_ref()
            .and_then(|e| hayai_opencl::SvmScratch::allocate(e, ffn_layer_bytes).ok())
    } else {
        None
    };
    println!(
        "  Transfer path: {:?}{}",
        if svm_scratch.is_some() {
            TransferPath::SvmZeroCopy
        } else {
            transfer_path
        },
        if svm_scratch.is_some() {
            " (SVM scratch active)"
        } else {
            ""
        }
    );

    let mut ping_pong = PingPongBuffer::new(ffn_layer_bytes);
    println!(
        "  Ping-Pong RAM: {} KB × 2 = {} KB",
        ffn_layer_bytes / 1024,
        ping_pong.total_allocated_bytes() / 1024
    );

    let mut pass_results = Vec::new();

    for pass in 1..=passes {
        let mut prefetcher =
            open_layer_reader(&tmp_path, ffn_layer_bytes, total_layers).await?;
        println!("  I/O backend:   {}", prefetcher.backend().as_str());

        let t_pass = Instant::now();
        let mut slot = 0usize;

        prefetcher.read_next_layer(ping_pong.prefetch_mut()).await?;
        ping_pong.swap();
        if let Some(ref mut svm) = svm_scratch {
            svm.ingest_layer(cl_engine.as_ref().unwrap(), slot, ping_pong.active())?;
        } else {
            scratch.ingest_layer(cl_engine.as_ref(), slot, ping_pong.active())?;
        }
        slot += 1;

        let mut layers_done = 0usize;
        let mut compute_secs = 0.0f64;

        while prefetcher.has_next() {
            let (active, prefetch) = ping_pong.active_and_prefetch_mut();
            let weights_q4 = &active[..tile_packed.min(active.len())];
            let prefetch_future = prefetcher.read_next_layer(prefetch);

            if weights_q4.len() == tile_packed {
                let t_c = Instant::now();
                if let Some(ref eng) = cl_engine {
                    if eng
                        .lut_matmul_q4(tile_m, tile_n, weights_q4, &lut, &input, &mut output)
                        .is_err()
                    {
                        cpu_lut_matmul_q4(tile_m, tile_n, weights_q4, &lut, &input, &mut output);
                    }
                } else {
                    cpu_lut_matmul_q4(tile_m, tile_n, weights_q4, &lut, &input, &mut output);
                }
                compute_secs += t_c.elapsed().as_secs_f64();
            }

            prefetch_future.await?;
            ping_pong.swap();
            if let Some(ref mut svm) = svm_scratch {
                svm.ingest_layer(cl_engine.as_ref().unwrap(), slot, ping_pong.active())?;
            } else {
                scratch.ingest_layer(cl_engine.as_ref(), slot, ping_pong.active())?;
            }
            slot += 1;
            layers_done += 1;
        }

        let active = ping_pong.active();
        let weights_q4 = &active[..tile_packed.min(active.len())];
        if weights_q4.len() == tile_packed {
            let t_c = Instant::now();
            cpu_lut_matmul_q4(tile_m, tile_n, weights_q4, &lut, &input, &mut output);
            compute_secs += t_c.elapsed().as_secs_f64();
        }
        layers_done += 1;

        if let Some(ref eng) = cl_engine {
            let _ = eng.queue.finish();
        }

        let pass_elapsed = t_pass.elapsed().as_secs_f64();
        let stats = prefetcher.stats(pass_elapsed);

        println!(
            "  Pass {pass}/{passes}: {:.2} MB in {:.3}s → {:.1} MB/s  (I/O stall: {:.1}%, compute: {:.3}s, layers: {layers_done})",
            stats.total_bytes as f64 / (1024.0 * 1024.0),
            stats.elapsed_secs,
            stats.throughput_mbps,
            stats.io_stall_fraction * 100.0,
            compute_secs,
        );
        pass_results.push(stats);
    }

    let avg_mbps =
        pass_results.iter().map(|s| s.throughput_mbps).sum::<f64>() / pass_results.len() as f64;
    let avg_stall = pass_results
        .iter()
        .map(|s| s.io_stall_fraction)
        .sum::<f64>()
        / pass_results.len() as f64;
    println!("─────────────────────────────────────────────────────────────");
    println!("  Average throughput:  {avg_mbps:.1} MB/s");
    println!(
        "  Average I/O stall:   {:.1}%  (lower = better overlap)",
        avg_stall * 100.0
    );
    if svm_scratch.is_some() {
        println!("  Note: SVM zero-copy path active (APU/unified memory)");
    }
    println!("─────────────────────────────────────────────────────────────");

    let _ = fs::remove_file(&tmp_path).await;
    Ok(())
}

fn parse_device_mode(device: &str) -> ExecutionMode {
    match device {
        "cpu" => ExecutionMode::CpuOnly,
        "auto" => ExecutionMode::Auto,
        other => ExecutionMode::OpenClDevice(other.to_string()),
    }
}

fn cmd_bench_hetero(
    device: String,
    tokens: usize,
    sinks: usize,
    window: usize,
    layers: usize,
) -> anyhow::Result<()> {
    let mode = parse_device_mode(&device);
    let mut config = ModelConfig::smollm_135m();
    if layers > 0 {
        config.num_layers = layers.min(config.num_layers);
    }

    let mut orch = EngineOrchestrator::new(mode, config.clone());

    println!("─────────────────────────────────────────────────────────────");
    println!("  HAYAI Heterogeneous Pipeline (Phase 3)");
    println!("  Model:     {}", config.name);
    println!("  Layers:    {}", config.num_layers);
    println!(
        "  Attn:      {} heads / {} KV (GQA), head_dim={}",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.hidden_size / config.num_attention_heads
    );
    println!("  KV Cache:  {sinks} sinks + window {window}");
    println!("  Tokens:    {tokens}");
    println!(
        "  Backend:   {}",
        if orch.using_opencl() {
            "CPU Attn + OpenCL FFN"
        } else {
            "CPU-Only"
        }
    );
    println!("─────────────────────────────────────────────────────────────");

    let (stats, hidden) =
        run_decode_benchmark(&mut orch, &config, tokens, sinks, window)?;

    let tok_per_s = tokens as f64 / stats.total_secs.max(1e-9);
    println!("  Decode wall:   {:.3}s  ({tok_per_s:.2} tok/s)", stats.total_secs);
    println!("  Attention CPU: {:.3}s", stats.attn_secs);
    println!("  FFN:           {:.3}s", stats.ffn_secs);
    println!("  Sync/overlap:  {:.3}s", stats.wait_secs);
    println!(
        "  Hidden L2:     {:.4}",
        hidden.iter().map(|x| x * x).sum::<f32>().sqrt()
    );

    println!("─────────────────────────────────────────────────────────────");
    println!("  Overlap probe (independent CPU Attn ∥ FFN)...");
    let p = run_overlap_probe(&mut orch, &config, 8);
    println!("  CPU alone:     {:.3}s", p.cpu_only_secs);
    println!(
        "  FFN alone:     {:.3}s ({})",
        p.gpu_only_secs,
        if p.used_opencl { "OpenCL" } else { "CPU" }
    );
    println!("  Parallel:      {:.3}s", p.parallel_secs);
    println!("  Speedup:       {:.2}x vs serial", p.speedup);
    println!("─────────────────────────────────────────────────────────────");
    Ok(())
}

// Prefer Q4_K_M (GGML type Q4_K) — fully supported by hayai-model GEMV.
const DEFAULT_SMOLLM_GGUF_URL: &str = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";

fn cmd_fetch_model(dir: PathBuf, url: Option<String>) -> anyhow::Result<()> {
    std::fs::create_dir_all(&dir)?;
    let url = url.unwrap_or_else(|| DEFAULT_SMOLLM_GGUF_URL.to_string());
    let filename = url
        .rsplit('/')
        .next()
        .unwrap_or("model.gguf")
        .to_string();
    let dest = dir.join(&filename);
    if dest.exists() {
        println!("Already present: {}", dest.display());
        return Ok(());
    }

    println!("Downloading\n  {url}\n→ {}", dest.display());
    let resp = ureq::get(&url)
        .set("User-Agent", "hayai/0.1")
        .call()
        .map_err(|e| anyhow::anyhow!("download failed: {e}"))?;
    let mut reader = resp.into_reader();
    let mut file = std::fs::File::create(&dest)?;
    let n = std::io::copy(&mut reader, &mut file)?;
    println!("Wrote {n} bytes to {}", dest.display());
    println!("Run: hayai-cli generate --model {}", dest.display());
    Ok(())
}

fn cmd_generate(
    model: PathBuf,
    prompt: String,
    max_tokens: usize,
    device: String,
    sinks: usize,
    window: usize,
    sample_name: String,
    temperature: f32,
    top_p: f32,
    seed: u64,
    use_chat: bool,
    dev_mmap: bool,
    memory_strategy: String,
) -> anyhow::Result<()> {
    let sampler = match sample_name.as_str() {
        "greedy" => SamplerConfig::Greedy,
        "temperature" => SamplerConfig::Temperature { temperature },
        "top_p" | "top-p" => SamplerConfig::TopP {
            temperature,
            top_p,
        },
        other => anyhow::bail!("unknown sampler '{other}' (greedy|temperature|top_p)"),
    };

    println!("─────────────────────────────────────────────────────────────");
    println!("  HAYAI Generate");
    println!("  Model:  {}", model.display());
    println!("  Prompt: {prompt:?}");
    println!("  ChatML: {use_chat}");
    println!(
        "  I/O:    {}",
        if dev_mmap {
            "DEV mmap (not PRD path)"
        } else {
            "deterministic stream (WeightIo → ping-pong scratch)"
        }
    );
    println!("  Memory: {}", memory_strategy);
    println!("─────────────────────────────────────────────────────────────");

    let rss_before = process_rss_bytes();
    let mode = parse_device_mode(&device);

    if dev_mmap {
        let t0 = Instant::now();
        let gguf = Arc::new(GgufFile::open(&model)?);
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let weights = LlamaWeights::load_arc(gguf)?;
        println!(
            "  Loaded {} in {:.2}s (mmap/dev, {:.1} MiB views)",
            weights.config.name,
            t0.elapsed().as_secs_f64(),
            weights.packed_nbytes as f64 / (1024.0 * 1024.0)
        );
        let prompt_text = if use_chat {
            tokenizer.apply_chat_template(&prompt)
        } else {
            prompt
        };
        let mut orch = EngineOrchestrator::new(mode, weights.config.clone());
        println!("  FFN device: {}", orch.ffn_device_name());
        let mut gen = Generator::new(weights, tokenizer, sinks, window, sampler, seed);
        let t1 = Instant::now();
        let stats = gen.generate(&mut orch, &prompt_text, max_tokens)?;
        print_gen_stats(&stats, t1.elapsed().as_secs_f64(), rss_before, process_rss_bytes());
        return Ok(());
    }

    let t0 = Instant::now();
    // Peek tokenizer via catalog open inside StreamingGenerator::open
    let cat = hayai_model::GgufCatalog::open(&model)?;
    let tokenizer = Tokenizer::from_catalog(&cat)?;
    drop(cat);
    let mut gen = StreamingGenerator::open(&model, tokenizer, sinks, window, sampler, seed)?;
    gen.set_memory_strategy(MemoryStrategy::parse(&memory_strategy));
    println!(
        "  Ready {} in {:.2}s (WeightIo={})",
        gen.config.name,
        t0.elapsed().as_secs_f64(),
        gen.io_backend.as_str()
    );
    let prompt_text = if use_chat {
        gen.tokenizer.apply_chat_template(&prompt)
    } else {
        prompt
    };
    let mut orch = EngineOrchestrator::new(mode, gen.config.clone());
    println!(
        "  FFN: {}{}",
        orch.ffn_device_name(),
        if orch.hetero_devices_active() {
            " + APU (up)"
        } else {
            ""
        }
    );
    let t1 = Instant::now();
    let stats = gen.generate(&mut orch, &prompt_text, max_tokens)?;
    let secs = t1.elapsed().as_secs_f64();
    print_gen_stats(&stats, secs, rss_before, process_rss_bytes());
    let serial = gen.attn_secs + gen.ffn_secs;
    let wall = gen.wall_compute_secs.max(secs);
    let speedup = if wall > 1e-9 { serial / wall } else { 1.0 };
    println!(
        "  I/O: {:.1} MiB via {} | scratch {:?} | io={:.2}s attn={:.2}s ffn={:.2}s dma={:.2}s map={:.2}s",
        gen.io_bytes as f64 / (1024.0 * 1024.0),
        gen.io_backend.as_str(),
        gen.transfer_path,
        gen.io_secs,
        gen.attn_secs,
        gen.ffn_secs,
        gen.dma_secs,
        gen.map_secs,
    );
    println!(
        "  Overlap: io∥ffn={:.2}s attn∥ffn_budget={:.2}s | wall={:.2}s serial_attn+ffn={:.2}s ratio={:.2}x | prefetch={} dGPU={} APU={}",
        gen.overlap_secs,
        gen.attn_ffn_overlap_secs,
        wall,
        serial,
        speedup,
        gen.prefetch_hits,
        gen.used_dgpu,
        gen.used_apu
    );
    if let Some(w) = gen.window_plan {
        println!(
            "  AdaptiveWindow: strategy={:?} k_chunk={} resident={} window={}",
            gen.memory_strategy,
            w.k_chunk,
            w.resident,
            format_bytes(w.window_bytes)
        );
    }
    if let Some(b) = gen.memory_budget {
        let rss = process_rss_bytes();
        println!(
            "  Budget:  ~{} (2×layer {} + KV {} + acts {})",
            format_bytes(b.total_budget_bytes),
            format_bytes(b.layer_window_bytes),
            format_bytes(b.kv_bytes),
            format_bytes(b.activation_bytes),
        );
        if let Some(r) = rss {
            let ratio = r as f64 / b.total_budget_bytes.max(1) as f64;
            println!(
                "  RSS vs budget: {} ({:.1}× budget; runtime/OpenCL/driver overhead expected)",
                format_bytes(r),
                ratio
            );
        }
    }
    Ok(())
}

fn print_gen_stats(
    stats: &hayai_core::GenerateStats,
    secs: f64,
    rss_before: Option<u64>,
    rss_after: Option<u64>,
) {
    println!("─────────────────────────────────────────────────────────────");
    println!("{}", stats.text);
    println!("─────────────────────────────────────────────────────────────");
    println!(
        "  prompt={} new={} in {:.2}s | decode {:.2} tok/s | wall {:.2} tok/s",
        stats.prompt_tokens,
        stats.new_tokens,
        secs,
        stats.new_tokens as f64 / secs.max(1e-9),
        stats.total_positions as f64 / secs.max(1e-9)
    );
    if let (Some(a), Some(b)) = (rss_before, rss_after) {
        println!(
            "  RSS: {} → {} (Δ {})",
            format_bytes(a),
            format_bytes(b),
            format_bytes(b.saturating_sub(a))
        );
    } else if let Some(b) = rss_after {
        println!("  RSS: {}", format_bytes(b));
    }
}

fn cmd_bench_generate(
    model: PathBuf,
    prompt: String,
    max_tokens: usize,
    device: String,
    sinks: usize,
    window: usize,
    use_chat: bool,
    memory_strategy: String,
) -> anyhow::Result<()> {
    println!("─────────────────────────────────────────────────────────────");
    println!("  HAYAI Bench Generate (Phase 5 — PRD streaming path)");
    println!("─────────────────────────────────────────────────────────────");

    let t0 = Instant::now();
    let cat = hayai_model::GgufCatalog::open(&model)?;
    let tokenizer = Tokenizer::from_catalog(&cat)?;
    let layer_bytes = cat.max_layer_pack_nbytes()?.max(1);
    drop(cat);
    let mut gen = StreamingGenerator::open(
        &model,
        tokenizer,
        sinks,
        window,
        SamplerConfig::Greedy,
        42,
    )?;
    gen.set_memory_strategy(MemoryStrategy::parse(&memory_strategy));
    let load_s = t0.elapsed().as_secs_f64();
    let prompt_text = if use_chat {
        gen.tokenizer.apply_chat_template(&prompt)
    } else {
        prompt.clone()
    };
    println!("  ChatML:        {use_chat}");
    let rss0 = process_rss_bytes();
    let budget = StreamingMemoryBudget::estimate(&gen.config, layer_bytes, sinks, window);

    let mode = parse_device_mode(&device);
    let mut orch = EngineOrchestrator::new(mode, gen.config.clone());
    println!("  FFN device:    {}", orch.ffn_device_name());
    if let Some(cl) = orch.opencl_engine() {
        println!(
            "  Transfer path: {:?}",
            select_transfer_path(&cl.device_info)
        );
        println!(
            "  Device VRAM:   {:.0} MiB (reported)",
            cl.device_info.global_mem_size as f64 / (1024.0 * 1024.0)
        );
    }
    println!(
        "  Hetero APU+GPU:{}",
        if orch.hetero_devices_active() {
            " yes"
        } else {
            " no (single accelerator or CPU)"
        }
    );
    println!(
        "  Budget:        ~{} (2×layer {} + KV {} + acts {})",
        format_bytes(budget.total_budget_bytes),
        format_bytes(budget.layer_window_bytes),
        format_bytes(budget.kv_bytes),
        format_bytes(budget.activation_bytes),
    );

    let t1 = Instant::now();
    let stats = gen.generate(&mut orch, &prompt_text, max_tokens)?;
    let secs = t1.elapsed().as_secs_f64();
    let rss1 = process_rss_bytes();
    let serial = gen.attn_secs + gen.ffn_secs;
    let wall = gen.wall_compute_secs.max(secs);
    let speedup = if wall > 1e-9 { serial / wall } else { 1.0 };

    println!("  Load time:     {load_s:.2}s (WeightIo={})", gen.io_backend.as_str());
    println!("  Scratch:       {:?}", gen.transfer_path);
    println!("  Prompt tokens: {}", stats.prompt_tokens);
    println!("  New tokens:    {}", stats.new_tokens);
    println!("  Wall time:     {secs:.2}s");
    println!(
        "  Decode tok/s:  {:.2}",
        stats.new_tokens as f64 / secs.max(1e-9)
    );
    println!(
        "  Prefill+dec:   {:.2} tok/s",
        stats.total_positions as f64 / secs.max(1e-9)
    );
    println!(
        "  Timers:        io={:.2}s attn={:.2}s ffn={:.2}s dma={:.2}s map={:.2}s",
        gen.io_secs, gen.attn_secs, gen.ffn_secs, gen.dma_secs, gen.map_secs
    );
    println!(
        "  Overlap:       io∥ffn={:.2}s attn∥ffn={:.2}s | ratio={:.2}x | prefetch={}",
        gen.overlap_secs, gen.attn_ffn_overlap_secs, speedup, gen.prefetch_hits
    );
    if let (Some(a), Some(b)) = (rss0, rss1) {
        println!(
            "  RSS:           {} → {} (Δ {}) | vs budget {:.1}×",
            format_bytes(a),
            format_bytes(b),
            format_bytes(b.saturating_sub(a)),
            b as f64 / budget.total_budget_bytes.max(1) as f64
        );
    } else if let Some(r) = rss1.or(rss0) {
        println!(
            "  RSS:           {} | vs budget {:.1}×",
            format_bytes(r),
            r as f64 / budget.total_budget_bytes.max(1) as f64
        );
    }
    println!(
        "  Sample out:    {:?}",
        stats.text.chars().take(80).collect::<String>()
    );
    if let Some(w) = gen.window_plan {
        println!(
            "  AdaptiveWindow: strategy={:?} k_chunk={} resident={} window={}",
            gen.memory_strategy,
            w.k_chunk,
            w.resident,
            format_bytes(w.window_bytes)
        );
    }
    if let Some(mem) = gen.owned_mem {
        println!(
            "  Hayai-owned:   peak {} (scratch={} kv={} acts={} staging={})",
            format_bytes(mem.peak_bytes),
            format_bytes(mem.scratch_bytes),
            format_bytes(mem.kv_bytes),
            format_bytes(mem.activation_bytes),
            format_bytes(mem.prefetch_staging_bytes),
        );
        let ok = mem.peak_bytes <= budget.total_budget_bytes;
        println!(
            "  Budget check:  {} (peak vs {})",
            if ok { "PASS" } else { "FAIL" },
            format_bytes(budget.total_budget_bytes)
        );
    }
    println!("─────────────────────────────────────────────────────────────");
    println!("  Compare: scripts/bench_compare.sh|.ps1 (tok/s + e2e + memory vs llama.cpp)");
    Ok(())
}
