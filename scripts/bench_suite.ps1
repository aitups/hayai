# Suite bench: SmolLM + HRM + Qwen3.5 + Gemma4 (hayai + llama.cpp side-by-side).
param(
    [string]$Prompt = "The capital of France is",
    [string]$Device = "auto",
    [int]$SmolTokens = 32,
    [int]$SuiteTokens = 8,
    [int]$LlamaPp = 5
)

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:Path = "C:\msys64\mingw64\bin;" +
    [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
    [System.Environment]::GetEnvironmentVariable("Path", "User")

$llamaBench = $null
$llamaCli = $null
$cmd = Get-Command llama-bench -ErrorAction SilentlyContinue
if ($cmd) { $llamaBench = $cmd.Source }
$cmd = Get-Command llama-cli -ErrorAction SilentlyContinue
if ($cmd) { $llamaCli = $cmd.Source }
$wingetPkg = "C:\Users\epoke\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe"
if (-not $llamaBench -and (Test-Path "$wingetPkg\llama-bench.exe")) {
    $llamaBench = "$wingetPkg\llama-bench.exe"
}
if (-not $llamaCli -and (Test-Path "$wingetPkg\llama-cli.exe")) {
    $llamaCli = "$wingetPkg\llama-cli.exe"
}

$suite = @(
    @{ Name = "SmolLM2-135M Q4_0"; Model = "models/SmolLM2-135M-Instruct-Q4_0.gguf"; N = $SmolTokens },
    @{ Name = "HRM-Text-1B Q5_K_M"; Model = "models/HRM-Text-1B-Q5_K_M.gguf"; N = $SuiteTokens },
    @{ Name = "Qwen3.5-4B Q4_K_M"; Model = "models/Qwen_Qwen3.5-4B-Q4_K_M.gguf"; N = $SuiteTokens },
    @{ Name = "Gemma4 E4B Q4_K_M"; Model = "models/google_gemma-4-E4B-it-Q4_K_M.gguf"; N = $SuiteTokens }
)

$outDir = Join-Path $Root "bench_results"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $outDir "suite_$stamp.log"
$summary = Join-Path $outDir "suite_$stamp.md"

function Log([string]$msg) {
    Write-Host $msg
    Add-Content -Path $log -Value $msg
}

function Pick-Match([string]$text, [string]$pattern) {
    if ($text -match $pattern) { return $Matches[1] }
    return "?"
}

function Clip-Sample([string]$s, [int]$max = 80) {
    $t = ($s -replace "\s+", " ").Trim()
    if ($t.Length -le $max) { return $t }
    return $t.Substring(0, $max) + "…"
}

Log "# Hayai suite bench $stamp"
Log "prompt=`"$Prompt`" device=$Device"
Log "llama-bench=$llamaBench"
Log "llama-cli=$llamaCli"
Log ""

$rows = New-Object System.Collections.Generic.List[object]

foreach ($m in $suite) {
    $path = Join-Path $Root $m.Model
    if (-not (Test-Path $path)) {
        Log "SKIP $($m.Name): missing $path"
        continue
    }
    $gib = [math]::Round((Get-Item $path).Length / 1GB, 3)
    Log "============================================================"
    Log "MODEL $($m.Name) ($gib GiB) n=$($m.N)"
    Log "============================================================"

    Log "--- hayai plan ---"
    cargo run -q -p hayai-cli -- plan --model $path 2>&1 | ForEach-Object {
        Log "$_"
    } | Out-Null

    Log "--- hayai bench-generate --raw ---"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $hayaiLines = cargo run -q -p hayai-cli --release -- bench-generate `
        --model $path --prompt $Prompt --max-tokens $m.N --device $Device --raw 2>&1
    $sw.Stop()
    $hayaiText = ($hayaiLines | ForEach-Object { "$_" }) -join "`n"
    $hayaiLines | ForEach-Object { Log "$_" }
    $e2e = [math]::Round($sw.Elapsed.TotalSeconds, 2)

    $decode = Pick-Match $hayaiText 'decode\s+([\d.]+)\s+tok/s'
    if ($decode -eq "?") { $decode = Pick-Match $hayaiText 'Decode tok/s:\s+([\d.]+)' }
    $owned = Pick-Match $hayaiText 'hayai_owned\s+~([\d.]+)\s*MiB'
    if ($owned -eq "?") { $owned = Pick-Match $hayaiText 'owned\s+~([\d.]+)' }
    $rss = Pick-Match $hayaiText 'RSS:.*?([\d.]+)\s*MiB\s*\('
    if ($rss -eq "?") { $rss = Pick-Match $hayaiText 'RSS:\s+([\d.]+)\s*MiB' }
    $dma = Pick-Match $hayaiText 'dma=([\d.]+)s'
    $ffn = Pick-Match $hayaiText 'ffn=([\d.]+)s'
    $attn = Pick-Match $hayaiText 'attn=([\d.]+)s'
    $io = Pick-Match $hayaiText 'io=([\d.]+)s'
    $hayaiSample = "?"
    if ($hayaiText -match 'Sample out:\s*"((?:\\.|[^"\\])*)"') {
        $hayaiSample = Clip-Sample ($Matches[1] -replace '\\n', ' ' -replace '\\"', '"')
    } elseif ($hayaiText -match 'Sample out:\s*(.+)') {
        $hayaiSample = Clip-Sample $Matches[1]
    } elseif ($hayaiText -match '(?s)Generated:\s*(.+?)(?:\r?\n---|$)') {
        $hayaiSample = Clip-Sample $Matches[1]
    }

    $lbPp = "n/a"
    $lbTg = "n/a"
    $llamaSample = "n/a"
    $llamaNote = ""

    if ($llamaBench) {
        Log "--- llama-bench -p $LlamaPp -n $($m.N) ---"
        $lbLines = & $llamaBench -m $path -p $LlamaPp -n $m.N 2>&1
        $lbText = ($lbLines | ForEach-Object { "$_" }) -join "`n"
        $lbLines | ForEach-Object { Log "$_" }
        if ($lbText -match "failed|error|unknown model|unsupported" -and $lbText -notmatch "pp$LlamaPp") {
            $llamaNote = "load/fail"
        }
        # llama-bench markdown/table: ppN and tgN columns
        if ($lbText -match ("pp" + $LlamaPp + "[^\d]*([\d.]+)")) {
            $lbPp = $Matches[1]
        }
        if ($lbText -match ("tg" + $m.N + "[^\d]*([\d.]+)")) {
            $lbTg = $Matches[1]
        } else {
            foreach ($line in $lbLines) {
                $s = "$line"
                if ($s -match '\|\s*[\w.-]+\s*\|' -and $s -match '([\d.]+)\s*\|\s*$') {
                    $lbTg = $Matches[1]
                }
            }
        }
    }

    if ($llamaCli) {
        Log "--- llama-cli sample (n=$($m.N), -no-cnv --temp 0) ---"
        $cliOut = Join-Path $outDir "llama_sample_$($m.Name -replace '[^\w.-]','_').txt"
        $cliErr = Join-Path $outDir "llama_sample_$($m.Name -replace '[^\w.-]','_').err"
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $llamaCli
            $psi.Arguments = "-m `"$path`" -n $($m.N) -p `"$Prompt`" -no-cnv --temp 0 --seed 42"
            $psi.UseShellExecute = $false
            $psi.RedirectStandardInput = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            $proc.StandardInput.Close()
            $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
            $stderrTask = $proc.StandardError.ReadToEndAsync()
            if (-not $proc.WaitForExit(180000)) {
                try { $proc.Kill() } catch {}
                $llamaNote = "cli-timeout"
                Log "llama-cli timeout - killed"
            }
            $raw = $stdoutTask.Result
            $err = $stderrTask.Result
            Set-Content -Path $cliOut -Value $raw -Encoding utf8
            Set-Content -Path $cliErr -Value $err -Encoding utf8
            Log "llama-cli exit=$($proc.ExitCode)"
            if ($raw) {
                Log $raw
                if ($raw -match ('(?s)>?\s*' + [regex]::Escape($Prompt) + '\r?\n(.+?)(?:\r?\n\s*\[ Prompt:|\r?\n\s*>|\z)')) {
                    $llamaSample = Clip-Sample $Matches[1]
                } elseif ($raw -match [regex]::Escape($Prompt) + '\s*(.+)') {
                    $llamaSample = Clip-Sample $Matches[1]
                } else {
                    $llamaSample = Clip-Sample (($raw -split "`n") | Select-Object -Last 5 | Out-String)
                }
            }
            if ($err -and $err -match "failed|error|unknown model|unsupported|not supported") {
                if ($llamaNote -eq "") { $llamaNote = "cli-fail" }
                Log $err
            }
        } catch {
            $llamaNote = "cli-exception"
            Log "llama-cli exception: $_"
        }
        Log "--- llama-cli sample done ($llamaSample) ---"
    }

    $rows.Add([pscustomobject]@{
        Model = $m.Name
        GiB = $gib
        N = $m.N
        Hayai_e2e_s = $e2e
        Hayai_decode_tps = $decode
        Hayai_owned_MiB = $owned
        RSS_MiB = $rss
        io_s = $io
        attn_s = $attn
        ffn_s = $ffn
        dma_s = $dma
        Hayai_sample = $hayaiSample
        llama_pp = $lbPp
        llama_tg = $lbTg
        llama_sample = $llamaSample
        llama_note = $llamaNote
    }) | Out-Null
    Log "--- row recorded for $($m.Name) ---"
}

Log ""
Log "## Summary: hayai vs llama.cpp"
$md = @(
    "# Hayai vs llama.cpp - suite $stamp",
    "",
    "Prompt (raw): $Prompt",
    "",
    "| model | GiB | n | hayai e2e_s | hayai decode t/s | llama pp$LlamaPp t/s | llama tg$SuiteTokens/n t/s | ratio tg/hayai | owned MiB |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
)
foreach ($r in $rows) {
    $ratio = "?"
    if ($r.Hayai_decode_tps -match '^[\d.]+$' -and $r.llama_tg -match '^[\d.]+$') {
        $ht = [double]$r.Hayai_decode_tps
        if ($ht -gt 0) {
            $ratio = ([math]::Round(([double]$r.llama_tg) / $ht, 1)).ToString([cultureinfo]::InvariantCulture) + "x"
        }
    }
    $md += "| $($r.Model) | $($r.GiB) | $($r.N) | $($r.Hayai_e2e_s) | $($r.Hayai_decode_tps) | $($r.llama_pp) | $($r.llama_tg) | $ratio | $($r.Hayai_owned_MiB) |"
}

$md += @(
    "",
    "## Samples (side-by-side)",
    "",
    "| model | hayai sample | llama-cli sample | note |",
    "| --- | --- | --- | --- |"
)
foreach ($r in $rows) {
    $hs = ($r.Hayai_sample -replace '\|', '/')
    $ls = ($r.llama_sample -replace '\|', '/')
    $md += "| $($r.Model) | $hs | $ls | $($r.llama_note) |"
}

$md += @(
    "",
    "## Timers (hayai)",
    "",
    "| model | io_s | attn_s | ffn_s | dma_s | RSS MiB |",
    "| --- | ---: | ---: | ---: | ---: | ---: |"
)
foreach ($r in $rows) {
    $md += "| $($r.Model) | $($r.io_s) | $($r.attn_s) | $($r.ffn_s) | $($r.dma_s) | $($r.RSS_MiB) |"
}

$md | ForEach-Object { Log $_ }
$md | Set-Content -Path $summary -Encoding utf8
Log ""
Log "Wrote $summary"
Log "Wrote $log"
Write-Host "SUMMARY_MD=$summary"
Write-Host "LOG=$log"
