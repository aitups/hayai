# Multi-metric Hayai vs llama.cpp compare (Windows host).
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Prompt = "Explain streaming weight inference in two sentences.",
    [int]$MaxTokens = 128,
    [string]$Device = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$ggufBytes = (Get-Item $Model).Length
$ggufGib = [math]::Round($ggufBytes / 1GB, 3)
Write-Host "=== hayai plan ==="
cargo run -q -p hayai-cli -- plan --model $Model
if ($LASTEXITCODE -ne 0) { throw "arch-blocked: unknown layer op" }

Write-Host "=== hayai bench-generate (--raw, greedy) ==="
$sw = [Diagnostics.Stopwatch]::StartNew()
cargo run -q -p hayai-cli --release -- bench-generate `
    --model $Model --prompt $Prompt --max-tokens $MaxTokens --device $Device --raw
$sw.Stop()
Write-Host "gguf_gib=$ggufGib e2e_wall_s=$([math]::Round($sw.Elapsed.TotalSeconds, 3))"

# Refresh PATH (winget installs often need a new shell)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

if (Get-Command llama-bench -ErrorAction SilentlyContinue) {
    Write-Host "=== llama-bench (pp512 / tg$MaxTokens) ==="
    llama-bench -m $Model -p 512 -n $MaxTokens
    Write-Host "=== llama-cli (e2e, temp=0) ==="
    $sw2 = [Diagnostics.Stopwatch]::StartNew()
    llama-cli -m $Model -p $Prompt -n $MaxTokens -c 4096 --temp 0 --top-k 1 --no-display-prompt -ngl 99
    $sw2.Stop()
    Write-Host "llama_cli_wall_s=$([math]::Round($sw2.Elapsed.TotalSeconds, 3))"
} elseif (Get-Command llama-cli -ErrorAction SilentlyContinue) {
    Write-Host "=== llama-cli ==="
    llama-cli -m $Model -p $Prompt -n $MaxTokens -c 4096 --temp 0 --top-k 1 -ngl 99
} else {
    Write-Host "llama.cpp not in PATH - winget install ggml.llamacpp"
}
