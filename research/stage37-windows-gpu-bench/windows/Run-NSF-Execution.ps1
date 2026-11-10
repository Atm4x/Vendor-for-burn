param(
    [int]$Iterations = 20,
    [int]$Warmup = 5,
    [string]$Frames = "64,256,1024",
    [ValidateSet("Both", "Parallel", "Sequential")]
    [string]$Mode = "Both"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "profile_nsf_execution.exe"
$checkpoint = Join-Path $PSScriptRoot "synthetic_nsf_decoder.pth"
$output = Join-Path $PSScriptRoot "nsf_cuda_execution.json"
$log = Join-Path $PSScriptRoot "nsf_cuda_execution.stderr.log"
$devices = Join-Path $PSScriptRoot "nsf_cuda_device.txt"
$archive = Join-Path $PSScriptRoot "NSF_CUDA_EXECUTION_RESULTS.zip"

if ($Iterations -lt 1 -or $Warmup -lt 0) { throw "Iterations must be positive and Warmup nonnegative" }
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing profiler EXE: $exe" }
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing synthetic checkpoint: $checkpoint" }

try {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Out-File -LiteralPath $devices -Encoding utf8
} catch {
    "nvidia-smi unavailable" | Out-File -LiteralPath $devices -Encoding utf8
}

$argsList = @(
    "--backend", "cuda",
    "--frames", $Frames,
    "--iterations", [string]$Iterations,
    "--warmup", [string]$Warmup,
    "--mode", $Mode.ToLowerInvariant(),
    "--checkpoint", "synthetic_nsf_decoder.pth"
)

Write-Host "NSF CUDA A/B: mode=$Mode; frames=$Frames; iterations=$Iterations; warmup=$Warmup"
$run = Start-Process -FilePath $exe -ArgumentList $argsList -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $output -RedirectStandardError $log -Wait -PassThru -NoNewWindow
if ($run.ExitCode -ne 0) {
    Get-Content -LiteralPath $log -ErrorAction SilentlyContinue
    throw "Profiler failed with exit code $($run.ExitCode)"
}

$result = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
if ($result.backend -ne "cuda" -or $result.cases.Count -lt 1) { throw "Invalid CUDA report" }
foreach ($case in $result.cases) {
    foreach ($variant in $case.variants) {
        Write-Host ("{0} frames; {1}: total={2:N4}ms; submit={3:N4}ms; readback={4:N4}ms; p90={5:N4}ms" -f $case.frames, $variant.mode, $variant.total_median_ms, $variant.submit_median_ms, $variant.readback_median_ms, $variant.total_p90_ms)
    }
    if ($Mode -eq "Both") {
        if ($null -eq $case.cross_mode_full_tensor_max_abs_error) { throw "Missing full tensor parity" }
        if ([double]$case.cross_mode_full_tensor_max_abs_error -gt 1.0e-5) { throw "Full tensor parity failed for $($case.frames) frames" }
        Write-Host ("{0} frames; sequential speedup={1:N3}x; max FP32 error={2:E}" -f $case.frames, $case.parallel_over_sequential_speedup, $case.cross_mode_full_tensor_max_abs_error)
    }
}

if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path @($output, $log, $devices) -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Results: $archive"