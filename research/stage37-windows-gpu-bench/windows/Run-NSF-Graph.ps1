param(
    [int]$Iterations = 20,
    [int]$Warmup = 5,
    [string]$Frames = "64,256,1024",
    [string]$Variants = "baseline,prepacked,packed,quad_interior",
    [ValidateSet("Sequential", "Parallel")]
    [string]$Execution = "Sequential"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "profile_nsf_graph_stage51.exe"
$checkpoint = Join-Path $PSScriptRoot "synthetic_nsf_decoder.pth"
$reports = Join-Path $PSScriptRoot "stage51_results"
$archive = Join-Path $PSScriptRoot "NSF_CUDA_GRAPH_STAGE51_RESULTS.zip"

if (-not (Test-Path -LiteralPath $exe)) { throw "Missing profiler: $exe" }
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
if ($Iterations -lt 1 -or $Warmup -lt 0) { throw "Invalid iteration count" }
New-Item -ItemType Directory -Force -Path $reports | Out-Null
try { & nvidia-smi -q > (Join-Path $reports "nvidia-smi.txt") 2>&1 } catch { $_ | Out-String | Set-Content (Join-Path $reports "nvidia-smi-error.txt") }

$all = @{
    baseline = @{ Kernel = ""; Prepack = $false }
    prepacked = @{ Kernel = ""; Prepack = $true }
    packed = @{ Kernel = "packed"; Prepack = $false }
    quad = @{ Kernel = "quad"; Prepack = $false }
    quad_interior = @{ Kernel = "quad_interior"; Prepack = $false }
}
$selected = @($Variants -split ',' | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
if ($selected.Count -eq 0 -or @($selected | Select-Object -Unique).Count -ne $selected.Count) { throw "Invalid or duplicate variants" }
foreach ($v in $selected) { if (-not $all.ContainsKey($v)) { throw "Unknown variant: $v" } }

$prevKernel = [Environment]::GetEnvironmentVariable("RVC_NSF_CONV_KERNEL", "Process")
$prevPrepack = [Environment]::GetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", "Process")
$runs = @()
try {
    for ($round = 1; $round -le 2; $round++) {
        $order = if ($round -eq 1) { $selected } else { @($selected[($selected.Count - 1)..0]) }
        foreach ($variant in $order) {
            $config = $all[$variant]
            if ($config.Kernel) { $env:RVC_NSF_CONV_KERNEL = $config.Kernel }
            else { Remove-Item Env:RVC_NSF_CONV_KERNEL -ErrorAction SilentlyContinue }
            if ($config.Prepack) { $env:RVC_CONV_WEIGHT_PREPACK = "1" }
            else { Remove-Item Env:RVC_CONV_WEIGHT_PREPACK -ErrorAction SilentlyContinue }
            $id = "round${round}_${variant}"
            $stdout = Join-Path $reports "$id.json"
            $stderr = Join-Path $reports "$id.stderr.log"
            $arguments = @(
                "--backend", "cuda",
                "--checkpoint", "synthetic_nsf_decoder.pth",
                "--execution", $Execution.ToLowerInvariant(),
                "--frames", $Frames,
                "--iterations", [string]$Iterations,
                "--warmup", [string]$Warmup
            )
            Write-Host "Running ${id}: ordinary NSF vs CUDA Graph Replay"
            $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -Wait -NoNewWindow
            if ($process.ExitCode -ne 0) {
                Get-Content $stderr -ErrorAction SilentlyContinue
                throw "$id failed with exit code $($process.ExitCode)"
            }
            $json = Get-Content $stdout -Raw | ConvertFrom-Json
            if ($json.backend -ne "cuda" -or @($json.cases).Count -eq 0) { throw "Unexpected JSON in $id" }
            foreach ($case in $json.cases) {
                if ($case.max_abs_error -gt 1.0e-5 -or $case.repeated_graph_max_abs_error -gt 1.0e-5) {
                    throw "FP32 parity failed for $id, $($case.frames) frames"
                }
                $label = if ($case.hardware_graph) { "HARDWARE CUDA GRAPH" } else { "FALLBACK, NOT A CUDA GRAPH" }
                Write-Host ("  frames={0}: ordinary={1:N4}ms, replay={2:N4}ms, ratio={3:N3}x, {4}, FP32={5:E}" -f $case.frames, $case.ordinary_median_ms, $case.replay_median_ms, $case.ordinary_over_replay, $label, $case.max_abs_error)
            }
            $runs += [pscustomobject]@{ id = $id; variant = $variant; report = $json }
        }
    }
}
finally {
    [Environment]::SetEnvironmentVariable("RVC_NSF_CONV_KERNEL", $prevKernel, "Process")
    [Environment]::SetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", $prevPrepack, "Process")
}

$summaryCases = @()
foreach ($run in $runs) {
    foreach ($case in $run.report.cases) {
        $summaryCases += [pscustomobject]@{
            id = $run.id
            variant = $run.variant
            frames = $case.frames
            hardware_graph = $case.hardware_graph
            capture_ms = $case.capture_setup_ms
            ordinary_ms = $case.ordinary_median_ms
            replay_ms = $case.replay_median_ms
            ordinary_dispatch_ms = $case.ordinary_dispatch_median_ms
            replay_dispatch_ms = $case.replay_dispatch_median_ms
            ordinary_readback_ms = $case.ordinary_readback_median_ms
            replay_readback_ms = $case.replay_readback_median_ms
            speedup = if ($case.hardware_graph) { $case.ordinary_over_replay } else { $null }
            parity_max_abs = $case.max_abs_error
            bitwise_equal = $case.exact_fp32_equal
        }
    }
}
$allHardware = @($summaryCases | Where-Object { -not $_.hardware_graph }).Count -eq 0
$summary = [pscustomobject]@{
    stage = 51
    device_backend = "cuda"
    static_input_only = $true
    execution = $Execution.ToLowerInvariant()
    iterations = $Iterations
    warmup = $Warmup
    has_hardware_graph_in_all_cases = $allHardware
    semantics = "Fixed input buffers for each shape; ordinary forward vs graph replay interleaved in one process, CPU readback included, graph capture amortization reported separately"
    cases = $summaryCases
}
$summary | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $reports "summary.json") -Encoding UTF8
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path (Join-Path $reports '*') -DestinationPath $archive -CompressionLevel Optimal
if (-not $allHardware) { Write-Warning "At least one capture fell back to ordinary closure replay. Do NOT interpret fallback ratios as CUDA Graph speedups." }
Write-Host "Finished: $archive"
