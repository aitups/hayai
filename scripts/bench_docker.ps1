# Build the Linux bench image and run hayai vs llama.cpp on the mounted models.
#
#   scripts\bench_docker.ps1
#   $env:STAGE=1; scripts\bench_docker.ps1                 # copy models to a Linux volume first
#   $env:MODELS="Qwen2.5-7B-Instruct-Q4_K_M.gguf"; $env:N=64; scripts\bench_docker.ps1
#
# Requires Docker Desktop.
#
# io_uring is blocked by Docker's default seccomp profile (io_uring_setup -> EPERM),
# so every run passes --security-opt seccomp=unconfined. Mounted models on Windows go
# through WSL2 drvfs (~200 MB/s), far below a real SSD — set STAGE=1 to copy the GGUFs
# onto the Linux volume (ext4) so I/O numbers reflect the disk instead of the mount.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$ModelsDir = if ($env:MODELS_DIR) { $env:MODELS_DIR } else { Join-Path $Root "models" }
$Image = if ($env:IMAGE) { $env:IMAGE } else { "hayai-bench-linux" }
$N = if ($env:N) { $env:N } else { "128" }
$PP = if ($env:PP) { $env:PP } else { "512" }
$OutDir = Join-Path $Root "bench_results"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

docker build -f (Join-Path $Root "scripts/bench_docker/Dockerfile") -t $Image $Root
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($env:STAGE -eq "1") {
    $Volume = if ($env:VOLUME) { $env:VOLUME } else { "hayai-bench-models" }
    docker volume create $Volume | Out-Null
    Write-Host "staging models into volume '$Volume' (native Linux fs)..."
    docker run --rm --entrypoint bash -v "${ModelsDir}:/src:ro" -v "${Volume}:/models" $Image `
        -c 'set -e; while read -r f; do [ -z "$f" ] && continue; [ -f "/models/$f" ] || cp "/src/$f" /models/; done < /usr/local/share/bench_models.txt; echo staged'
    $Mount = "${Volume}:/models"
} else {
    $Mount = "${ModelsDir}:/models:ro"
}

$runArgs = @("--rm", "--security-opt", "seccomp=unconfined", "-v", $Mount, "-e", "N=$N", "-e", "PP=$PP")
if ($env:THREADS) { $runArgs += @("-e", "THREADS=$($env:THREADS)") }
if ($env:MODELS) { $runArgs += @("-e", "MODELS=$($env:MODELS)") }
if ($env:MEMORY_STRATEGY) { $runArgs += @("-e", "MEMORY_STRATEGY=$($env:MEMORY_STRATEGY)") }

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $OutDir "bench_linux_$stamp.md"
docker run @runArgs $Image | Tee-Object -FilePath $out
Write-Host "`nsaved: $out"
