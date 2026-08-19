//! HuggingFace model resolution + download (`--hf <repo>` / `--hf-file <name>`).
//!
//! Defaults to the first `*Q4_K_M.gguf` in the repo; `--hf-file` overrides the
//! exact GGUF file. Files are cached in the models directory.

use std::path::Path;

fn api(repo: &str) -> anyhow::Result<serde_json::Value> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let resp = ureq::get(&url)
        .set("User-Agent", "hayai/0.1")
        .call()
        .map_err(|e| anyhow::anyhow!("HF api call failed for {repo}: {e}"))?;
    Ok(serde_json::from_reader(resp.into_reader())?)
}

fn repo_files(repo: &str) -> anyhow::Result<Vec<String>> {
    let json = api(repo)?;
    let siblings = json
        .get("siblings")
        .and_then(|s| s.as_array())
        .ok_or_else(|| anyhow::anyhow!("HF api for {repo} has no siblings"))?;
    Ok(siblings
        .iter()
        .filter_map(|s| s.get("rfilename").and_then(|v| v.as_str()).map(str::to_owned))
        .collect())
}

/// Resolve the GGUF file name to download for a repo.
pub fn resolve_hf_file(repo: &str, hf_file: Option<&str>) -> anyhow::Result<String> {
    let files = repo_files(repo)?;
    if let Some(file) = hf_file {
        if files.iter().any(|f| f == file) {
            return Ok(file.to_string());
        }
        anyhow::bail!(
            "file '{file}' not found in {repo}; available: {}",
            files
                .iter()
                .filter(|f| f.ends_with(".gguf"))
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    if let Some(q4km) = files.iter().find(|f| f.ends_with("Q4_K_M.gguf")) {
        return Ok(q4km.clone());
    }
    let ggufs: Vec<&String> = files.iter().filter(|f| f.ends_with(".gguf")).collect();
    if ggufs.is_empty() {
        anyhow::bail!("repo {repo} contains no GGUF files");
    }
    let names: Vec<&str> = ggufs.iter().map(|s| s.as_str()).collect();
    anyhow::bail!("no *Q4_K_M.gguf in {repo}; available: {}", names.join(", "))
}

/// Download `file` from `repo` into `dest_dir` (skips if already present).
pub fn download(repo: &str, file: &str, dest_dir: &Path) -> anyhow::Result<std::path::PathBuf> {
    let dest = dest_dir.join(file);
    if dest.exists() {
        return Ok(dest);
    }
    std::fs::create_dir_all(dest_dir)?;
    let url = format!("https://huggingface.co/{repo}/resolve/main/{file}");
    println!("Downloading {url}\n→ {}", dest.display());
    let resp = ureq::get(&url)
        .set("User-Agent", "hayai/0.1")
        .call()
        .map_err(|e| anyhow::anyhow!("download failed: {e}"))?;
    let mut reader = resp.into_reader();
    let mut out = std::fs::File::create(&dest)?;
    std::io::copy(&mut reader, &mut out)?;
    Ok(dest)
}

/// Resolve + download the model file; returns the local path.
pub fn download_model(
    repo: &str,
    hf_file: Option<&str>,
    dest_dir: &Path,
) -> anyhow::Result<std::path::PathBuf> {
    let file = resolve_hf_file(repo, hf_file)?;
    download(repo, &file, dest_dir)
}
