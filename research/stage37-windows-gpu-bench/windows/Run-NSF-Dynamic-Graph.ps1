param(
    [int]$Iterations = 16,
    [string]$Frames = "64,256,1024",
    [string]$Variants = "prepacked,baseline"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "profile_nsf_graph_stage52.exe"
$checkpoint = Join-Path $PSScriptRoot "synthetic_nsf_decoder.pth"
$reports = Join-Path $PSScriptRoot "stage52_results"
$archive = Join-Path $PSScriptRoot "NSF_CUDA_DYNAMIC_GRAPH_STAGE52_RESULTS.zip"

if (-not (Test-Path -LiteralPath $exe)) { throw "Missing profiler: $exe" }
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
if ($Iterations -lt 4) { throw "At least four distinct inputs are required" }
$variantsArray = @($Variants -split ',' | ForEach-Object { $_.Trim().ToLowerInvariant() } | Where-Object { $_ })
if ($variantsArray.Count -eq 0 -or @($variantsArray | Select-Object -Unique).Count -ne $variantsArray.Count) {
    throw "Empty or duplicate variants"
}
foreach ($v in $variantsArray) {
    if ($v -notin @("baseline", "prepacked")) { throw "Unsupported variant: $v" }
}

New-Item -ItemType Directory -Force -Path $reports | Out-Null
try { & nvidia-smi -q > (Join-Path $reports "nvidia-smi.txt") 2>&1 }
catch { $_ | Out-String | Set-Content (Join-Path $reports "nvidia-smi-error.txt") }

$previousPrepack = [Environment]::GetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", "Process")
$previousKernel = [Environment]::GetEnvironmentVariable("RVC_NSF_CONV_KERNEL", "Process")
$results = @()
try {
    for ($round = 1; $round -le 2; $round++) {
        $order = if ($round -eq 1) { $variantsArray } else { @($variantsArray[($variantsArray.Count - 1)..0]) }
        foreach ($variant in $order) {
            Remove-Item Env:RVC_NSF_CONV_KERNEL -ErrorAction SilentlyContinue
            if ($variant -eq "prepacked") { $env:RVC_CONV_WEIGHT_PREPACK = "1" }
            else { Remove-Item Env:RVC_CONV_WEIGHT_PREPACK -ErrorAction SilentlyContinue }
            $id = "round${round}_${variant}"
            $stdout = Join-Path $reports "$id.json"
            $stderr = Join-Path $reports "$id.stderr.log"
            $arguments = @(
                "--backend", "cuda",
                "--checkpoint", "synthetic_nsf_decoder.pth",
                "--execution", "sequential",
                "--frames", $Frames,
                "--iterations", [string]$Iterations
            )
            Write-Host "Running $id: dynamic CUDA Graph input refresh"
            $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -Wait -NoNewWindow
            if ($process.ExitCode -ne 0) {
                Get-Content $stderr -ErrorAction SilentlyContinue
                throw "$id failed, exit code $($process.ExitCode)"
            }
            $report = Get-Content -LiteralPath $stdout -Raw | ConvertFrom-Json
            if ($report.stage -ne 52 -or $report.backend -ne "cuda" -or -not $report.basic_refresh_passed -or -not $report.basic_graph_hardware) {
                throw "Basic dynamic CUDA Graph capture/refresh failed for $id"
            }
            foreach ($case in $report.cases) {
                if (-not $case.hardware_graph -or -not $case.bitwise_equal -or $case.max_abs_error -gt 1.0e-5 -or $case.distinct_outputs -lt 2) {
                    throw "Dynamic CUDA Graph proof failed for $id / $($case.frames) frames"
                }
                Write-Host ("  frames={0} refresh={1:N4}ms replay={2:N4}ms ordinary={3:N4}ms distinct={4} max_error={5:E}" -f $case.frames, $case.median_refresh_ms, $case.median_replay_ms, $case.median_normal_ms, $case.distinct_outputs, $case.max_abs_error)
            }
            $results += [pscustomobject]@{ id = $id; variant = $variant; report = $report }
        }
    }
}
finally {
    [Environment]::SetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", $previousPrepack, "Process")
    [Environment]::SetEnvironmentVariable("RVC_NSF_CONV_KERNEL", $previousKernel, "Process")
}

$summary = [pscustomobject]@{
    stage = 52
    experiment = "persistent-buffer-dynamic-cuda-graph"
    backend = "cuda"
    input_changes_between_replays = $true
    source_changes_between_replays = $true
    conditioning_changes_between_replays = $true
    rounds = 2
    iterations = $Iterations
    all_cases_passed = $true
    cases = @($results | ForEach-Object {
        $id = $_.id
        $variant = $_.variant
        $_.report.cases | ForEach-Object {
            [pscustomobject]@{
                id = $id
                variant = $variant
                frames = $_.frames
                hardware_graph = $_.hardware_graph
                bitwise_equal = $_.bitwise_equal
                max_abs_error = $_.max_abs_error
                distinct_outputs = $_.distinct_outputs
                capture_ms = $_.capture_setup_ms
                refresh_ms = $_.median_refresh_ms
                replay_ms = $_.median_replay_ms
                ordinary_ms = $_.median_normal_ms
            }
        }
    })
}
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $reports "summary.json") -Encoding UTF8
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path (Join-Path $reports '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Completed: $archive"
