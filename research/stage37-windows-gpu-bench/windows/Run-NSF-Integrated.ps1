param(
    [int]$Iterations = 20,
    [int]$Warmup = 5,
    [string]$Frames = "64,256,1024",
    [ValidateSet("cuda", "wgpu", "cpu")]
    [string]$Backend = "cuda"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "profile_nsf_prepack_stage49.exe"
$checkpoint = Join-Path $PSScriptRoot "synthetic_nsf_decoder.pth"
$reports = Join-Path $PSScriptRoot "stage50_results"
$archive = Join-Path $PSScriptRoot "NSF_INTEGRATED_STAGE50_RESULTS.zip"
if (-not (Test-Path -LiteralPath $exe)) { throw "Missing benchmark executable: $exe" }
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing benchmark checkpoint: $checkpoint" }
New-Item -ItemType Directory -Force -Path $reports | Out-Null

$variants = @(
    [pscustomobject]@{ Name = "baseline_parallel"; Execution = "parallel"; Kernel = ""; Prepacked = $false },
    [pscustomobject]@{ Name = "sequential"; Execution = "sequential"; Kernel = ""; Prepacked = $false },
    [pscustomobject]@{ Name = "stage49_prepacked"; Execution = "sequential"; Kernel = ""; Prepacked = $true },
    [pscustomobject]@{ Name = "stage50_packed"; Execution = "sequential"; Kernel = "packed"; Prepacked = $false },
    [pscustomobject]@{ Name = "stage50_quad"; Execution = "sequential"; Kernel = "quad"; Prepacked = $false },
    [pscustomobject]@{ Name = "stage50_quad_interior"; Execution = "sequential"; Kernel = "quad_interior"; Prepacked = $false }
)

$oldKernel = [Environment]::GetEnvironmentVariable("RVC_NSF_CONV_KERNEL", "Process")
$oldPrepack = [Environment]::GetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", "Process")
$oldTrace = [Environment]::GetEnvironmentVariable("RVC_NSF_CONV_TRACE", "Process")
$runs = @()
try {
    $orders = @($variants, @($variants[($variants.Count - 1)..0]))
    for ($round = 0; $round -lt $orders.Count; $round++) {
        foreach ($variant in $orders[$round]) {
            $id = "round$($round + 1)_$($variant.Name)"
            if ($variant.Kernel) { $env:RVC_NSF_CONV_KERNEL = $variant.Kernel }
            else { Remove-Item Env:RVC_NSF_CONV_KERNEL -ErrorAction SilentlyContinue }
            if ($variant.Prepacked) { $env:RVC_CONV_WEIGHT_PREPACK = "1" }
            else { Remove-Item Env:RVC_CONV_WEIGHT_PREPACK -ErrorAction SilentlyContinue }
            Remove-Item Env:RVC_NSF_CONV_TRACE -ErrorAction SilentlyContinue
            $stdout = Join-Path $reports "$id.json"
            $stderr = Join-Path $reports "$id.stderr.log"
            $args = @(
                "--backend", $Backend,
                "--checkpoint", "synthetic_nsf_decoder.pth",
                "--execution", $variant.Execution,
                "--frames", $Frames,
                "--iterations", [string]$Iterations,
                "--warmup", [string]$Warmup,
                "--dump-prefix", (Join-Path $reports $id)
            )
            Write-Host "[$id] $Backend / $($variant.Execution) / $($variant.Kernel)"
            $p = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -Wait -NoNewWindow
            if ($p.ExitCode -ne 0) {
                Get-Content $stderr -ErrorAction SilentlyContinue
                throw "$id failed with exit code $($p.ExitCode)"
            }
            $value = Get-Content -LiteralPath $stdout -Raw | ConvertFrom-Json
            if ($value.backend -ne $Backend -or $value.cases.Count -lt 1) { throw "Invalid result $id" }
            if (@($value.cases | Where-Object { $_.repeated_max_abs_error -gt 0.00001 }).Count -ne 0) { throw "Non-deterministic results in $id" }
            $runs += [pscustomobject]@{ id = $id; variant = $variant.Name; report = $value }
        }
    }
}
finally {
    [Environment]::SetEnvironmentVariable("RVC_NSF_CONV_KERNEL", $oldKernel, "Process")
    [Environment]::SetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", $oldPrepack, "Process")
    [Environment]::SetEnvironmentVariable("RVC_NSF_CONV_TRACE", $oldTrace, "Process")
}

$cases = @()
foreach ($rawFrame in ($Frames -split ',')) {
    $frame = [int]$rawFrame
    $first = Join-Path $reports "round1_baseline_parallel-$frame.f32le"
    $bytes = [IO.File]::ReadAllBytes($first)
    $asFloats = [float[]]::new($bytes.Length / 4)
    [Buffer]::BlockCopy($bytes, 0, $asFloats, 0, $bytes.Length)
    foreach ($run in $runs) {
        $file = Join-Path $reports "$($run.id)-$frame.f32le"
        $actual = [IO.File]::ReadAllBytes($file)
        if ($actual.Length -ne $bytes.Length) { throw "Tensor size mismatch: $file" }
        $maxError = 0.0
        for ($i = 0; $i -lt $asFloats.Length; $i++) {
            $v = [BitConverter]::ToSingle($actual, $i * 4)
            if ([float]::IsNaN($v) -or [float]::IsInfinity($v)) { throw "Non-finite FP32 in $file" }
            $maxError = [Math]::Max($maxError, [Math]::Abs([double]$v - [double]$asFloats[$i]))
        }
        if ($maxError -gt 0.00001) { throw "FP32 mismatch $($run.id) ($frame frames), max_abs=$maxError" }
        $timing = @($run.report.cases | Where-Object { $_.frames -eq $frame })[0]
        $cases += [pscustomobject]@{
            frames = $frame
            variant = $run.variant
            run_id = $run.id
            median_total_ms = $timing.median_total_ms
            median_submit_ms = $timing.median_submit_ms
            median_readback_ms = $timing.median_readback_ms
            load_ms = $run.report.load_ms
            max_abs_error = $maxError
        }
    }
}
$summary = [pscustomobject]@{
    stage = 50
    backend = $Backend
    iterations = $Iterations
    warmup = $Warmup
    semantics = "Full synthetic NSF forward; CPU readback included; 2 process-order-balanced rounds; fixed public checkpoint; cross-variant full FP32 parity"
    results = $cases
}
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $reports "summary.json") -Encoding UTF8
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path (Join-Path $reports '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Completed. Send back: $archive"
