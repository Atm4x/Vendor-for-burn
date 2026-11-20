param(
    [int]$Iterations = 20,
    [int]$Warmup = 5,
    [string]$Frames = "64,256,1024",
    [ValidateSet("Sequential", "Parallel")]
    [string]$Execution = "Sequential"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "profile_nsf_prepack_stage49.exe"
$checkpoint = Join-Path $PSScriptRoot "synthetic_nsf_decoder.pth"
$results = Join-Path $PSScriptRoot "reports"
$archive = Join-Path $PSScriptRoot "NSF_PREPACK_CUDA_RESULTS.zip"

if (-not (Test-Path -LiteralPath $exe)) { throw "Missing profiler: $exe" }
if (-not (Test-Path -LiteralPath $checkpoint)) { throw "Missing synthetic checkpoint: $checkpoint" }
New-Item -ItemType Directory -Force -Path $results | Out-Null

try { & nvidia-smi -q > (Join-Path $results "nvidia-smi.txt") 2>&1 } catch { $_ | Out-String | Set-Content (Join-Path $results "nvidia-smi-error.txt") }
try { & nvcc --version > (Join-Path $results "nvcc-version.txt") 2>&1 } catch { $_ | Out-String | Set-Content (Join-Path $results "nvcc-error.txt") }

$hadPrevious = Test-Path Env:RVC_CONV_WEIGHT_PREPACK
$previous = if ($hadPrevious) { $env:RVC_CONV_WEIGHT_PREPACK } else { $null }
$runs = @()
try {
    foreach ($item in @(
        @{ Id = "baseline1"; Prepack = $false },
        @{ Id = "prepacked1"; Prepack = $true },
        @{ Id = "prepacked2"; Prepack = $true },
        @{ Id = "baseline2"; Prepack = $false }
    )) {
        $id = $item.Id
        if ($item.Prepack) { $env:RVC_CONV_WEIGHT_PREPACK = "1" }
        else { Remove-Item Env:RVC_CONV_WEIGHT_PREPACK -ErrorAction SilentlyContinue }
        $json = Join-Path $results "$id.json"
        $stderr = Join-Path $results "$id.stderr.log"
        $prefix = Join-Path $results $id
        $arguments = @(
            "--backend", "cuda",
            "--checkpoint", "synthetic_nsf_decoder.pth",
            "--execution", $Execution.ToLowerInvariant(),
            "--frames", $Frames,
            "--iterations", [string]$Iterations,
            "--warmup", [string]$Warmup,
            "--dump-prefix", $prefix
        )
        Write-Host "Running $id; execution=$Execution; frames=$Frames"
        $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $json -RedirectStandardError $stderr -Wait -PassThru -NoNewWindow
        if ($process.ExitCode -ne 0) {
            Get-Content $stderr -ErrorAction SilentlyContinue
            throw "$id failed with exit code $($process.ExitCode)"
        }
        $value = Get-Content $json -Raw | ConvertFrom-Json
        if ($value.backend -ne "cuda" -or $value.cases.Count -lt 1) { throw "Invalid JSON output for $id" }
        $expected = if ($item.Prepack) { "cached_weight_layout" } else { "baseline" }
        if ($value.preparation -ne $expected) { throw "$id did not use expected prepack mode" }
        $runs += [pscustomobject]@{ id = $id; data = $value }
    }
}
finally {
    if ($hadPrevious) { $env:RVC_CONV_WEIGHT_PREPACK = $previous }
    else { Remove-Item Env:RVC_CONV_WEIGHT_PREPACK -ErrorAction SilentlyContinue }
}

function Compare-Fp32([string]$Left, [string]$Right) {
    $leftBytes = [System.IO.File]::ReadAllBytes($Left)
    $rightBytes = [System.IO.File]::ReadAllBytes($Right)
    if ($leftBytes.Length -ne $rightBytes.Length -or ($leftBytes.Length % 4) -ne 0) { throw "Output tensor length mismatch" }
    $maxError = 0.0
    $bitwise = $true
    for ($i = 0; $i -lt $leftBytes.Length; $i += 4) {
        if ($leftBytes[$i] -ne $rightBytes[$i] -or $leftBytes[$i+1] -ne $rightBytes[$i+1] -or $leftBytes[$i+2] -ne $rightBytes[$i+2] -or $leftBytes[$i+3] -ne $rightBytes[$i+3]) { $bitwise = $false }
        $a = [System.BitConverter]::ToSingle($leftBytes, $i)
        $b = [System.BitConverter]::ToSingle($rightBytes, $i)
        if ([single]::IsNaN($a) -or [single]::IsNaN($b)) { throw "NaN in FP32 comparison" }
        $maxError = [Math]::Max($maxError, [Math]::Abs([double]$a - [double]$b))
    }
    return [pscustomobject]@{ max_abs_error = $maxError; bitwise_equal = $bitwise; elements = $leftBytes.Length / 4 }
}

$cases = @()
foreach ($frameCount in ($Frames -split ',')) {
    $frameCount = [int]$frameCount
    $comparison = Compare-Fp32 (Join-Path $results "baseline1-$frameCount.f32le") (Join-Path $results "prepacked1-$frameCount.f32le")
    $comparison2 = Compare-Fp32 (Join-Path $results "baseline2-$frameCount.f32le") (Join-Path $results "prepacked2-$frameCount.f32le")
    $maxError = [Math]::Max($comparison.max_abs_error, $comparison2.max_abs_error)
    if ($maxError -gt 1.0e-5) { throw "Full FP32 parity failed for $frameCount frames: $maxError" }
    $baseline1 = @($runs[0].data.cases | Where-Object frames -eq $frameCount)[0]
    $prepacked1 = @($runs[1].data.cases | Where-Object frames -eq $frameCount)[0]
    $prepacked2 = @($runs[2].data.cases | Where-Object frames -eq $frameCount)[0]
    $baseline2 = @($runs[3].data.cases | Where-Object frames -eq $frameCount)[0]
    $speedup1 = [double]$baseline1.median_total_ms / [double]$prepacked1.median_total_ms
    $speedup2 = [double]$baseline2.median_total_ms / [double]$prepacked2.median_total_ms
    $cases += [pscustomobject]@{
        frames = $frameCount
        baseline_medians_ms = @($baseline1.median_total_ms, $baseline2.median_total_ms)
        prepacked_medians_ms = @($prepacked1.median_total_ms, $prepacked2.median_total_ms)
        pair_speedups = @($speedup1, $speedup2)
        full_fp32_max_abs_error = $maxError
        bitwise_equal = ($comparison.bitwise_equal -and $comparison2.bitwise_equal)
    }
    Write-Host ("{0}: baseline {1:N4}/{2:N4}ms, prepacked {3:N4}/{4:N4}ms, speedup {5:N3}/{6:N3}x, FP32 error {7:E}" -f $frameCount, $baseline1.median_total_ms, $baseline2.median_total_ms, $prepacked1.median_total_ms, $prepacked2.median_total_ms, $speedup1, $speedup2, $maxError)
}
$summary = [pscustomobject]@{
    stage = 49
    backend = "cuda"
    execution = $Execution.ToLowerInvariant()
    frames = $Frames
    iterations = $Iterations
    warmup = $Warmup
    model_load_ms = $runs | ForEach-Object { [pscustomobject]@{ id = $_.id; load_ms = $_.data.load_ms } }
    cases = $cases
    semantics = "Two warm independent Baseline/Prepack process pairs; model-load time excludes device and checkpoint initialization; full output parity after run"
}
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $results "summary.json") -Encoding UTF8
if (Test-Path $archive) { Remove-Item $archive -Force }
Compress-Archive -Path (Join-Path $results '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Completed. Send back: $archive"
