param(
    [int]$Iterations = 12,
    [int]$Warmup = 4,
    [string]$Frames = "64,256,1024"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "profile_nsf_stage42.exe"
$fixture = Join-Path $PSScriptRoot "synthetic_nsf_decoder.pth"
$output = Join-Path $PSScriptRoot "nsf_cuda_scheduling.json"
$log = Join-Path $PSScriptRoot "nsf_cuda_scheduling.stderr.log"

if (-not (Test-Path -LiteralPath $exe)) { throw "Missing profiler EXE: $exe" }
if (-not (Test-Path -LiteralPath $fixture)) { throw "Missing synthetic checkpoint: $fixture" }

$arguments = @(
    "--backend", "cuda",
    "--frames", $Frames,
    "--iterations", [string]$Iterations,
    "--warmup", [string]$Warmup,
    "--checkpoint", $fixture
)

Write-Host "Running NSF CUDA scheduling A/B: $Frames frames; $Iterations iterations; $Warmup warmup"
$run = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $output -RedirectStandardError $log -Wait -PassThru -NoNewWindow
if ($run.ExitCode -ne 0) {
    Get-Content -LiteralPath $log -ErrorAction SilentlyContinue
    throw "NSF CUDA profiler exited with code $($run.ExitCode)"
}

$result = Get-Content -LiteralPath $output -Raw | ConvertFrom-Json
if ($result.backend -ne "cuda" -or $result.cases.Count -eq 0) { throw "Invalid profiler output" }
foreach ($case in $result.cases) {
    if ($case.max_abs_error -gt 1.0e-5) { throw "FP32 parity failed at $($case.frames) frames" }
    Write-Host ("{0} frames: default={1:N3} flush-pair={2:N3} flush-conv={3:N3} sync-pair={4:N3} ms; max_err={5:E}" -f $case.frames, $case.zero_seeded_ms, $case.flush_per_pair_ms, $case.flush_per_conv_ms, $case.sync_per_pair_ms, $case.max_abs_error)
}
Write-Host "JSON: $output"
Write-Host "Diagnostics: $log"
