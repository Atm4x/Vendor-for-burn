param(
    [int]$Iterations = 20,
    [int]$Warmup = 5,
    [string]$Frames = "64,256,1024"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "profile_nsf_prepack_stage49.exe"
$fixture = Join-Path $PSScriptRoot "synthetic_nsf_decoder.pth"
$reports = Join-Path $PSScriptRoot "stage51_streams"
$archive = Join-Path $PSScriptRoot "NSF_CUDA_STREAMS_STAGE51_RESULTS.zip"
if (-not (Test-Path $exe) -or -not (Test-Path $fixture)) { throw "Missing profiler or synthetic checkpoint" }
if ($Iterations -lt 1 -or $Warmup -lt 0) { throw "Invalid iterations/warmup" }
New-Item -ItemType Directory -Force -Path $reports | Out-Null

$configs = @(
    [pscustomobject]@{ Name = "perthread_parallel"; Policy = "per-thread"; MaxStreams = 128; Execution = "parallel" },
    [pscustomobject]@{ Name = "single_parallel"; Policy = "single"; MaxStreams = 1; Execution = "parallel" },
    [pscustomobject]@{ Name = "perthread_sequential"; Policy = "per-thread"; MaxStreams = 128; Execution = "sequential" },
    [pscustomobject]@{ Name = "single_sequential"; Policy = "single"; MaxStreams = 1; Execution = "sequential" }
)

$previousPrepack = [Environment]::GetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", "Process")
$previousKernel = [Environment]::GetEnvironmentVariable("RVC_NSF_CONV_KERNEL", "Process")
$runs = @()
try {
    $env:RVC_CONV_WEIGHT_PREPACK = "1"
    Remove-Item Env:RVC_NSF_CONV_KERNEL -ErrorAction SilentlyContinue
    for ($round = 1; $round -le 2; $round++) {
        $order = if ($round -eq 1) { $configs } else { @($configs[($configs.Count - 1)..0]) }
        foreach ($config in $order) {
            $id = "round${round}_$($config.Name)"
            $work = Join-Path $reports $id
            New-Item -ItemType Directory -Force -Path $work | Out-Null
            Copy-Item -LiteralPath $fixture -Destination (Join-Path $work "synthetic_nsf_decoder.pth") -Force
            $toml = "[streaming]`npolicy = `"$($config.Policy)`"`nmax_streams = $($config.MaxStreams)`n"
            Set-Content -LiteralPath (Join-Path $work "cubecl.toml") -Value $toml -Encoding ascii
            $stdout = Join-Path $work "result.json"
            $stderr = Join-Path $work "stderr.log"
            $args = @(
                "--backend", "cuda",
                "--checkpoint", "synthetic_nsf_decoder.pth",
                "--execution", $config.Execution,
                "--frames", $Frames,
                "--iterations", [string]$Iterations,
                "--warmup", [string]$Warmup,
                "--dump-prefix", "nsf-output"
            )
            Write-Host "[$id] CUDA streams=$($config.Policy), max_streams=$($config.MaxStreams), NSF=$($config.Execution)"
            $process = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $work -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru -NoNewWindow
            if ($process.ExitCode -ne 0) {
                Get-Content $stderr -ErrorAction SilentlyContinue
                throw "Profiler failed ($id): $($process.ExitCode)"
            }
            $result = Get-Content -LiteralPath $stdout -Raw | ConvertFrom-Json
            if ($result.backend -ne "cuda" -or $result.preparation -ne "cached_weight_layout" -or @($result.cases).Count -eq 0) {
                throw "Wrong profiler mode: $id"
            }
            $runs += [pscustomobject]@{ id = $id; config = $config; work = $work; data = $result }
        }
    }
}
finally {
    [Environment]::SetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", $previousPrepack, "Process")
    [Environment]::SetEnvironmentVariable("RVC_NSF_CONV_KERNEL", $previousKernel, "Process")
}

$reference = $runs[0]
$summaryCases = @()
foreach ($run in $runs) {
    foreach ($case in $run.data.cases) {
        $frames = [int]$case.frames
        $referencePath = Join-Path $reference.work "nsf-output-$frames.f32le"
        $observedPath = Join-Path $run.work "nsf-output-$frames.f32le"
        $left = [IO.File]::ReadAllBytes($referencePath)
        $right = [IO.File]::ReadAllBytes($observedPath)
        if ($left.Length -ne $right.Length -or ($left.Length % 4) -ne 0) { throw "Tensor length mismatch at $frames" }
        $bitwiseEqual = $true
        $maxError = 0.0
        for ($i = 0; $i -lt $left.Length; $i += 4) {
            $a = [BitConverter]::ToSingle($left, $i)
            $b = [BitConverter]::ToSingle($right, $i)
            if ([single]::IsNaN($b) -or [single]::IsInfinity($b)) { throw "Nonfinite output at $frames" }
            if ([BitConverter]::ToInt32($left, $i) -ne [BitConverter]::ToInt32($right, $i)) { $bitwiseEqual = $false }
            $maxError = [Math]::Max($maxError, [Math]::Abs([double]$a - [double]$b))
        }
        if ($maxError -gt 1e-5) { throw "FP32 parity failed: $($run.id), frames=$frames, max_abs=$maxError" }
        $summaryCases += [pscustomobject]@{
            id = $run.id
            policy = $run.config.Policy
            max_streams = $run.config.MaxStreams
            execution = $run.config.Execution
            frames = $frames
            median_ms = $case.median_total_ms
            submit_ms = $case.median_submit_ms
            readback_ms = $case.median_readback_ms
            bitwise_equal = $bitwiseEqual
            max_abs_error = $maxError
        }
        Write-Host ("  {0} frames: {1:N4}ms (FP32 max_abs {2:E})" -f $frames, $case.median_total_ms, $maxError)
    }
}
$summary = [pscustomobject]@{
    stage = 51
    experiment = "cuda-native-stream-policy"
    variant = "Stage49 Prepacked NSF (fixed shape synthetic checkpoint)"
    iterations = $Iterations
    warmup = $Warmup
    semantics = "Two reverse-order rounds, one fresh CUDA process per configuration; per-thread vs single stream, parallel vs sequential residual branches; complete CPU readback and FP32 parity"
    cases = $summaryCases
}
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $reports "summary.json") -Encoding UTF8
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path (Join-Path $reports '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Finished: $archive"
