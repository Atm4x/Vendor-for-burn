param(
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(1,800)][int]$CoreFrames = 300,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateSet('all','baseline','nsf','gru','both','baseline-and-both')][string]$Modes = 'all'
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($item in @($exe,$Voice,$Hubert,$Rmvpe,$Index,$InputWav)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw "Missing file: $item" }
}
$exe = (Resolve-Path -LiteralPath $exe).Path
$Voice = (Resolve-Path -LiteralPath $Voice).Path
$Hubert = (Resolve-Path -LiteralPath $Hubert).Path
$Rmvpe = (Resolve-Path -LiteralPath $Rmvpe).Path
$Index = (Resolve-Path -LiteralPath $Index).Path
$InputWav = (Resolve-Path -LiteralPath $InputWav).Path
$root = Join-Path $PSScriptRoot 'stage68_profiler_results'
$archive = Join-Path $PSScriptRoot 'RVC_STAGE68_PROFILE_RESULTS.zip'
if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
New-Item -ItemType Directory -Path $root -Force | Out-Null
$allCases = @(
    [pscustomobject]@{name='baseline';gemm=$false;persistent=$false},
    [pscustomobject]@{name='nsf';gemm=$true;persistent=$false},
    [pscustomobject]@{name='gru';gemm=$false;persistent=$true},
    [pscustomobject]@{name='both';gemm=$true;persistent=$true}
)
$cases = @($allCases | Where-Object {
    $Modes -eq 'all' -or $_.name -eq $Modes -or
    ($Modes -eq 'baseline-and-both' -and $_.name -in @('baseline','both'))
})
if ($cases.Count -eq 0) { throw 'No profiling modes selected' }
$keys = @(
    'RVC_PROFILE','RVC_DIAGNOSTIC_TRACE','RVC_CONV_WEIGHT_PREPACK',
    'RVC_NSF_CONV_KERNEL','RVC_NSF_IMPLICIT_GEMM','RVC_RMVPE_GRU_PERSISTENT',
    'RVC_GPU_FAISS','RVC_GPU_F0','RVC_GLOBAL_F0',
    'RVC_RMVPE_FUSED_READBACK','RVC_RMVPE_COMPACT_READBACK','RVC_RMVPE_PLANE_READBACK'
)
$previous = @{}
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$process = $null
$results = @()
try {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    $env:RVC_PROFILE = '1'
    $env:RVC_DIAGNOSTIC_TRACE = '1'
    $env:RVC_CONV_WEIGHT_PREPACK = '1'
    $nvidia = Get-Command nvidia-smi.exe -ErrorAction Stop
    & $nvidia.Source -q | Out-File -LiteralPath (Join-Path $root 'nvidia-before.txt') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA driver unavailable' }
    & $exe doctor | Out-File -LiteralPath (Join-Path $root 'doctor.txt') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'Unified RVC doctor failed' }
    Get-FileHash -Algorithm SHA256 -LiteralPath $exe,$Voice,$Hubert,$Rmvpe,$Index,$InputWav |
        Select-Object Path,Hash | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $root 'inputs-sha256.json') -Encoding UTF8
    [pscustomobject]@{
        stage=68; time_utc=(Get-Date).ToUniversalTime().ToString('o')
        modes=$Modes; core_frames=$CoreFrames; tensor_batch_size=$TensorBatchSize
        backend='cuda'; cpu_faiss=$true; gpu_f0=$false; global_f0=$false
        prepacked_weights=$true; cuda_graph=$false
        profiler='synchronized RVC_TRACE spans, Windows process RSS/CPU and NVIDIA GPU utilization/VRAM/power/SM clocks'
        rust_huge_profiler='Debian 13 Linux perf/Heaptrack/Tracy bundle validated separately; not a Windows CUDA profiler'
        measure='one representative-shape warmup followed by one full synchronized diagnostic per mode'
        caution='diagnostic stage spans are nested and introduce sync overhead; do not sum parent and child spans or compare directly to Stage67 wall-clock benchmark'
    } | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $root 'configuration.json') -Encoding UTF8
    foreach ($case in $cases) {
        Remove-Item Env:RVC_NSF_IMPLICIT_GEMM,Env:RVC_RMVPE_GRU_PERSISTENT -ErrorAction SilentlyContinue
        if ($case.gemm) { $env:RVC_NSF_IMPLICIT_GEMM = '1' }
        if ($case.persistent) { $env:RVC_RMVPE_GRU_PERSISTENT = '1' }
        $folder = Join-Path $root $case.name
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
        $wav = Join-Path $folder 'converted-full.wav'
        $out = Join-Path $folder 'benchmark.stdout.log'
        $err = Join-Path $folder 'benchmark.stderr.log'
        $telemetry = Join-Path $folder 'gpu_process_telemetry.csv'
        $argsArray = @('infer-wav-full-diagnostic',$Voice,$Hubert,$Rmvpe,$Index,$InputWav,$wav,
            'cuda',[string]$CoreFrames,[string]$TensorBatchSize)
        Write-Host "Stage68 $($case.name): synchronized full WAV profiling with prepack=1, gemm=$($case.gemm), persistent=$($case.persistent)"
        $psi = [System.Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $exe
        $psi.WorkingDirectory = $PSScriptRoot
        $psi.UseShellExecute = $false
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.Arguments = (@($argsArray | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }) -join ' ')
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $psi
        if (-not $process.Start()) { throw "Could not start profiler mode $($case.name)" }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        'timestamp_utc,rss_mb,cpu_total_ms,gpu_util_pct,gpu_memory_util_pct,vram_used_mb,power_w,sm_clock_mhz,gpu_temp_c' |
            Set-Content -LiteralPath $telemetry -Encoding UTF8
        while (-not $process.WaitForExit(750)) {
            $stamp = (Get-Date).ToUniversalTime().ToString('o')
            $proc = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
            $rss = if ($proc) { [Math]::Round($proc.WorkingSet64/1MB,2) } else { '' }
            $cpu = if ($proc) { [Math]::Round($proc.TotalProcessorTime.TotalMilliseconds,2) } else { '' }
            $gpu = ''
            try {
                $gpu = (& $nvidia.Source '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu' '--format=csv,noheader,nounits' 2>$null | Select-Object -First 1)
            } catch {}
            $parts = if ($gpu -and $gpu.Split(',').Count -ge 6) {
                @($gpu.Split(',') | ForEach-Object { $_.Trim() })
            } else { @('','','','','','') }
            ('"{0}",{1},{2},{3},{4},{5},{6},{7},{8}' -f $stamp,$rss,$cpu,$parts[0],$parts[1],$parts[2],$parts[3],$parts[4],$parts[5]) |
                Add-Content -LiteralPath $telemetry -Encoding UTF8
        }
        $process.WaitForExit()
        $stdoutTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $out -Encoding UTF8
        $stderrTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $err -Encoding UTF8
        $exit = $process.ExitCode
        $process.Dispose()
        $process = $null
        if ($exit -ne 0) {
            Get-Content -LiteralPath $err -Tail 40 | Out-Host
            throw "Stage68 $($case.name): executable returned code $exit"
        }
        $events = @(
            Get-Content -LiteralPath $err | Where-Object { $_.StartsWith('RVC_TRACE ') } |
                ForEach-Object { $_.Substring(10) | ConvertFrom-Json }
        )
        $spans = @($events | Where-Object { $_.type -eq 'span' -and $_.phase -eq 'measured' })
        if ($spans.Count -lt 80) { throw "Stage68 $($case.name): expected >=80 measured RVC_TRACE spans, found $($spans.Count)" }
        $events | ForEach-Object { $_ | ConvertTo-Json -Depth 15 -Compress } |
            Set-Content -LiteralPath (Join-Path $folder 'stages.jsonl') -Encoding UTF8
        $summary = @($spans | Group-Object name | ForEach-Object {
            $members = @($_.Group)
            [pscustomobject]@{
                name = $_.Name
                calls = $members.Count
                sum_total_ms = [Math]::Round((($members | Measure-Object total_ms -Sum).Sum),3)
                average_ms = [Math]::Round((($members | Measure-Object total_ms -Average).Average),3)
                sum_enqueue_and_cpu_ms = [Math]::Round((($members | Measure-Object enqueue_and_cpu_ms -Sum).Sum),3)
                sum_sync_ms = [Math]::Round((($members | Measure-Object sync_ms -Sum).Sum),3)
            }
        } | Sort-Object sum_total_ms -Descending)
        $summary | Export-Csv -LiteralPath (Join-Path $folder 'stage-time-summary.csv') -NoTypeInformation -Encoding UTF8
        @($spans | Where-Object { $_.name -eq 'nsf.upsample_resblocks' -or $_.name.StartsWith('nsf.stage_') } |
            ForEach-Object { [pscustomobject]@{
                span=$_.name; batch_id=$_.batch_id; stage=$_.details.stage; total_ms=$_.total_ms;
                cpu_ms=$_.enqueue_and_cpu_ms; sync_ms=$_.sync_ms;
            } }) | Export-Csv -LiteralPath (Join-Path $folder 'nsf-by-stage.csv') -NoTypeInformation -Encoding UTF8
        $reportPath = "$wav.diagnostic.json"
        if (-not (Test-Path -LiteralPath $reportPath)) { throw "Stage68 $($case.name): missing detailed benchmark JSON" }
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        if ($report.stage -ne 64 -or $report.backend -ne 'cuda' -or $report.measured_full_passes -ne 1 -or
            $report.source_seconds -lt 60 -or $report.chunk_count -lt 10 -or $report.output_samples -lt 100000) {
            throw "Stage68 $($case.name): incomplete full-WAV diagnostic"
        }
        $results += [pscustomobject]@{
            mode=$case.name; gemm=$case.gemm; persistent=$case.persistent
            load_ms=$report.load_ms; warmup_ms=$report.warmup_ms
            full_inference_ms=$report.full_inference_ms; diagnostic_rtf=$report.rtf
            chunks=$report.chunk_count; output_samples=$report.output_samples
            output_sha256=(Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash
            measured_spans=$spans.Count
        }
        Write-Host ('  diagnostic full={0:N2}s, {1} spans, output={2}' -f ($report.full_inference_ms/1000),$spans.Count,$report.output_samples)
    }
    [pscustomobject]@{
        stage=68; experiment='synchronized-stage-profiling-with-four-algorithm-modes'; cases=$results;
        warning='Diagnostic full-inference values include CUDA sync at nested stages and are not comparable to uninstrumented RTF; parent span totals overlap their children'
    } | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    if ($null -ne $process) {
        if (-not $process.HasExited) { $process.Kill() }
        $process.Dispose()
    }
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $archive -CompressionLevel Optimal
    Write-Host "Stage68 diagnostic ZIP: $archive"
}
