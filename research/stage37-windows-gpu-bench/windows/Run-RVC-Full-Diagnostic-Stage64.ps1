param(
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(1, 800)][int]$CoreFrames = 300,
    [ValidateRange(1, 32)][int]$TensorBatchSize = 2
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($item in @($exe,$Voice,$Hubert,$Rmvpe,$Index,$InputWav)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw "Missing file: $item" }
}

$results = Join-Path $PSScriptRoot 'stage64_diagnostic_results'
$archive = Join-Path $PSScriptRoot 'RVC_STAGE64_FULL_DIAGNOSTIC_RESULTS.zip'
if (Test-Path -LiteralPath $results) { Remove-Item $results -Recurse -Force }
New-Item -ItemType Directory -Force -Path $results | Out-Null
$stdout = Join-Path $results 'benchmark.stdout.log'
$stderr = Join-Path $results 'benchmark.stderr.log'
$wav = Join-Path $results 'converted-full.wav'
$telemetry = Join-Path $results 'gpu_process_telemetry.csv'
$trace = Join-Path $results 'stages.jsonl'
$previous = @{}
$keys = @('RVC_PROFILE','RVC_DIAGNOSTIC_TRACE','RVC_GPU_FAISS','RVC_GPU_F0','RVC_RMVPE_COMPACT_READBACK','RVC_RMVPE_FUSED_READBACK','RVC_RMVPE_PLANE_READBACK','RVC_CONV_WEIGHT_PREPACK')
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$process = $null
try {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    $env:RVC_PROFILE = '1'
    $env:RVC_DIAGNOSTIC_TRACE = '1'
    $env:RVC_CONV_WEIGHT_PREPACK = '1'
    $command = Get-Command nvidia-smi.exe -ErrorAction Stop
    & $command.Source -q | Out-File -LiteralPath (Join-Path $results 'nvidia-driver-and-device.txt') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'NVIDIA driver not available' }
    & $exe doctor | Out-File -LiteralPath (Join-Path $results 'rvc-doctor.txt') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'Unified EXE doctor failed' }
    Get-FileHash -Algorithm SHA256 -LiteralPath $exe,$Voice,$Hubert,$Rmvpe,$Index,$InputWav |
        Select-Object Path,Hash | ConvertTo-Json -Depth 3 |
        Set-Content -LiteralPath (Join-Path $results 'file-sha256.json') -Encoding UTF8
    [pscustomobject]@{
        timestamp=(Get-Date).ToUniversalTime().ToString('o')
        os=[Environment]::OSVersion.VersionString
        cpu=[Environment]::ProcessorCount
        powershell=$PSVersionTable.PSVersion.ToString()
        core_frames=$CoreFrames
        tensor_batch_size=$TensorBatchSize
        backend='cuda'
        faiss='cpu IVF nprobe=1, index_rate=0.75'
        rmvpe='CPU FP64 F0, dense readback'
        prepacked=$true
        cuda_graph=$false
        benchmark_passes=1
        warmup='one representative-shape batch; excluded from measured full WAV'
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $results 'run-configuration.json') -Encoding UTF8
    $culture=[Globalization.CultureInfo]::InvariantCulture
    $argumentList=@('infer-wav-full-diagnostic', $Voice, $Hubert, $Rmvpe, $Index, $InputWav, $wav,
            'cuda', $CoreFrames.ToString($culture), $TensorBatchSize.ToString($culture))
    $quoted=@($argumentList | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
    Write-Host 'Stage64: ONE full WAV, ONE CUDA baseline, detailed synchronized trace...'
    $process=Start-Process -FilePath $exe -ArgumentList $quoted -WorkingDirectory $PSScriptRoot `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -NoNewWindow
    'timestamp_utc,working_set_mb,cpu_total_ms,gpu_util_pct,gpu_memory_util_pct,vram_used_mb,power_w,sm_clock_mhz,gpu_temp_c' |
        Set-Content -LiteralPath $telemetry -Encoding UTF8
    while (-not $process.HasExited) {
        $stamp=(Get-Date).ToUniversalTime().ToString('o')
        $proc=Get-Process -Id $process.Id -ErrorAction SilentlyContinue
        $rss=if ($proc) { [Math]::Round($proc.WorkingSet64 / 1MB,2) } else { '' }
        $cpu=if ($proc) { [Math]::Round($proc.TotalProcessorTime.TotalMilliseconds,2) } else { '' }
        $gpu=''
        try {
            $gpu=(& $command.Source '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu' '--format=csv,noheader,nounits' 2>$null | Select-Object -First 1)
        } catch {}
        if ($gpu -and $gpu.Split(',').Length -ge 6) {
            $parts=@($gpu.Split(',') | ForEach-Object { $_.Trim() })
        } else { $parts=@('','','','','','') }
        ('"{0}",{1},{2},{3},{4},{5},{6},{7},{8}' -f $stamp,$rss,$cpu,$parts[0],$parts[1],$parts[2],$parts[3],$parts[4],$parts[5]) |
            Add-Content -LiteralPath $telemetry -Encoding UTF8
        Start-Sleep -Milliseconds 750
        $process.Refresh()
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        Get-Content -LiteralPath $stderr -Tail 35 | Out-Host
        throw "Stage64 full CUDA diagnostic failed, exit $($process.ExitCode)"
    }
    $events = @(
        Get-Content -LiteralPath $stderr | Where-Object { $_.StartsWith('RVC_TRACE ') } | ForEach-Object {
            $_.Substring(10) | ConvertFrom-Json
        }
    )
    if ($events.Count -lt 30) { throw "Expected detailed trace, found only $($events.Count) records" }
    $events | ForEach-Object { $_ | ConvertTo-Json -Depth 12 -Compress } |
        Set-Content -LiteralPath $trace -Encoding UTF8
    $spans=@($events | Where-Object { $_.type -eq 'span' -and $_.phase -eq 'measured' })
    $spans | Group-Object name | ForEach-Object {
        $items=$_.Group
        [pscustomobject]@{
            name=$_.Name
            calls=$items.Count
            sum_total_ms=[Math]::Round((($items | Measure-Object total_ms -Sum).Sum),3)
            mean_total_ms=[Math]::Round((($items | Measure-Object total_ms -Average).Average),3)
            sum_sync_ms=[Math]::Round((($items | Measure-Object sync_ms -Sum).Sum),3)
        }
    } | Sort-Object sum_total_ms -Descending | Export-Csv -NoTypeInformation -LiteralPath (Join-Path $results 'stage-time-summary.csv') -Encoding UTF8
    $reportPath="$wav.diagnostic.json"
    if (-not (Test-Path $wav) -or -not (Test-Path $reportPath)) { throw 'Missing complete output or diagnostic summary' }
    $report=Get-Content $reportPath -Raw | ConvertFrom-Json
    if ($report.stage -ne 64 -or $report.backend -ne 'cuda' -or $report.measured_full_passes -ne 1 -or $report.source_seconds -lt 60 -or $report.chunk_count -lt 10) {
        throw 'Diagnostic validation failed: benchmark was not a single full CUDA WAV'
    }
    $report | ConvertTo-Json -Depth 12 | Set-Content (Join-Path $results 'single-full-wav-summary.json') -Encoding UTF8
    Write-Host ("Full WAV {0:N2}s, measured {1:N2}s, diagnostic RTF {2:N4}, chunks {3}, trace spans {4}" -f `
        $report.source_seconds,($report.full_inference_ms/1000),$report.rtf,$report.chunk_count,$spans.Count)
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    if ($null -ne $process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    if (Test-Path $archive) { Remove-Item $archive -Force }
    Compress-Archive -Path (Join-Path $results '*') -DestinationPath $archive -CompressionLevel Optimal
    Write-Host "Results archive: $archive"
}
