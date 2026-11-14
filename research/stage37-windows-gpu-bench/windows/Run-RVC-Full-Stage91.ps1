param(
    [ValidateSet('cuda','cpu','wgpu')][string]$Device = 'cuda',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(2,16)][int]$WarmRuns = 2,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(1,916)][int]$CoreFrames = 900,
    [ValidateRange(0,3)][int]$ExtraChunks = 1,
    [ValidateRange(0,1)][double]$IndexRate = 0.75,
    [ValidateSet('all','baseline','phase')][string]$Modes = 'all'
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($path in @($exe,$Voice,$Hubert,$Rmvpe,$InputWav)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
if ($IndexRate -gt 0 -and ($Index -eq '-' -or -not (Test-Path -LiteralPath $Index -PathType Leaf))) {
    throw 'IndexRate > 0 requires an existing .index file'
}
if ($CoreFrames + 84 -gt 1000) {
    throw 'A model segment may not exceed 1000 frames including context and trailing frames'
}
$root = Join-Path $PSScriptRoot 'stage91_full_results'
$zip = Join-Path $PSScriptRoot 'RVC_STAGE91_FULL_WAV_RESULTS.zip'
if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
[void](New-Item -ItemType Directory -Path $root -Force)
$changes = @(
    'RVC_CHUNK_STRATEGY', 'RVC_QUIET_EXTRA_CHUNKS', 'RVC_NSF_GLOBAL_PHASE', 'RVC_NSF_GLOBAL_NOISE',
    'RVC_NSF_RESBLOCK_FP16', 'RVC_NSF_RESBLOCK_FP16_MIN_CHANNELS',
    'RVC_NSF_PAIR_SHARED_TILE', 'RVC_NSF_TILE_CORE_FRAMES',
    'RVC_NSF_IMPLICIT_GEMM', 'RVC_RMVPE_GRU_PERSISTENT',
    'RVC_RMVPE_MEL_REUSE', 'RVC_GLOBAL_F0', 'RVC_GPU_F0',
    'RVC_GPU_FAISS', 'RVC_PROFILE', 'RVC_DIAGNOSTIC_TRACE',
    'RVC_RMVPE_UNET_COMPENSATED_TF32', 'RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32'
)
$previous = @{}
foreach ($key in $changes) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$cases = @(
    [pscustomobject]@{ name='baseline_1'; group='baseline'; phase='0' },
    [pscustomobject]@{ name='phase_1'; group='phase'; phase='1' },
    [pscustomobject]@{ name='phase_2'; group='phase'; phase='1' },
    [pscustomobject]@{ name='baseline_2'; group='baseline'; phase='0' }
)
$selected = @($cases | Where-Object { $Modes -eq 'all' -or $_.group -eq $Modes })
$results = @()
$failures = @()
$culture = [Globalization.CultureInfo]::InvariantCulture
try {
    foreach ($key in $changes) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    $env:RVC_NSF_IMPLICIT_GEMM = '1'
    $env:RVC_RMVPE_GRU_PERSISTENT = '1'
    $env:RVC_GLOBAL_F0 = '1'
    if ($Device -eq 'cuda') {
        try { & nvidia-smi -q > (Join-Path $root 'nvidia-smi.txt') 2>&1 }
        catch { $_ | Out-String | Set-Content (Join-Path $root 'nvidia-smi-error.txt') }
    }
    foreach ($case in $selected) {
        $env:RVC_CHUNK_STRATEGY = 'quiet-splice'
        $env:RVC_QUIET_EXTRA_CHUNKS = [string]$ExtraChunks
        $env:RVC_NSF_GLOBAL_PHASE = $case.phase
        $wav = Join-Path $root "$($case.name).wav"
        $stdout = Join-Path $root "$($case.name).stdout.log"
        $stderr = Join-Path $root "$($case.name).stderr.log"
        $arguments = @('infer-wav-full-benchmark',$Voice,$Hubert,$Rmvpe,$Index,$InputWav,$wav,
            $IndexRate.ToString($culture),$Device,[string]$WarmRuns,[string]$CoreFrames,[string]$TensorBatchSize)
        $quoted = @($arguments | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
        Write-Host "Stage91 $($case.name): global_phase=$($case.phase) strategy=quiet-splice extra=$ExtraChunks core=$CoreFrames batch=$TensorBatchSize"
        $startInfo = New-Object Diagnostics.ProcessStartInfo
        $startInfo.FileName = $exe
        $startInfo.Arguments = ($quoted -join ' ')
        $startInfo.WorkingDirectory = $PSScriptRoot
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $process = New-Object Diagnostics.Process
        $process.StartInfo = $startInfo
        try {
            try {
                if (-not $process.Start()) { throw 'Process failed to start' }
                $outTask = $process.StandardOutput.ReadToEndAsync()
                $errTask = $process.StandardError.ReadToEndAsync()
                $process.WaitForExit()
                [IO.File]::WriteAllText($stdout,$outTask.GetAwaiter().GetResult(),[Text.Encoding]::UTF8)
                [IO.File]::WriteAllText($stderr,$errTask.GetAwaiter().GetResult(),[Text.Encoding]::UTF8)
                if ($process.ExitCode -ne 0) { throw "Exit code $($process.ExitCode)" }
                $report = Get-Content -LiteralPath "${wav}.benchmark.json" -Raw | ConvertFrom-Json
                if ($report.stage -ne 63 -or $report.backend -ne $Device -or
                    $report.chunk_strategy -ne 'quiet-splice' -or
                    $report.planned_chunk_count -ne $report.chunk_counts[0] -or
                    [int]$report.core_frames -ne [int]$CoreFrames -or
                    $report.planned_model_input_seconds -le 0 -or
                    $report.inference_ms.Count -ne $WarmRuns -or
                    [bool]$report.rmvpe_mel_reuse -ne $false -or
                    [bool]$report.global_f0 -ne $true -or
                    [bool]$report.nsf_global_phase -ne ($case.phase -eq '1')) {
                    throw "Invalid Stage91 benchmark metadata for $($case.name)"
                }
                $result = [pscustomobject]@{
                    name=$case.name
                    group=$case.group
                    strategy='quiet-splice'
                    global_phase=$case.phase
                    extra_chunks=$ExtraChunks
                    core_frames=$CoreFrames
                    model_input_seconds=$report.planned_model_input_seconds
                    cut_frames=$report.planned_cut_frames
                    chunks=$report.planned_chunk_count
                    source_seconds=$report.source_seconds
                    warm_median_ms=$report.warm_median_ms
                    first_inference_ms=$report.first_inference_ms
                    warm_rtf=$report.warm_rtf
                    output_samples=$report.output_samples
                    iteration_stable=$report.iterations_exact_f32
                    sha256=(Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash
                }
                $results += $result
                Write-Host ('  warm={0:N1}ms, source={1:N2}s, modelInput={2:N2}s, chunks={3}, stable={4}' -f
                    $result.warm_median_ms,$result.source_seconds,$result.model_input_seconds,$result.chunks,$result.iteration_stable)
            } catch {
                $message = "$($case.name): $($_.Exception.Message)"
                Write-Warning $message
                $failures += $message
                $message | Set-Content (Join-Path $root "$($case.name).failure.txt")
            }
        } finally { $process.Dispose() }
    }
    foreach ($group in @('baseline','phase')) {
        $pair = @($results | Where-Object { $_.group -eq $group })
        if ($pair.Count -eq 2 -and $pair[0].sha256 -ne $pair[1].sha256) {
            $failures += "Non-deterministic Stage91 pair ${group}: WAV SHA256 differs"
        }
    }
    if ($results.Count -ne $selected.Count) {
        $failures += "Incomplete comparison: $($results.Count)/$($selected.Count) successful cases"
    }
    [pscustomobject]@{
        stage=91
        experiment='global-f0-conditioned-nsf-phase'
        backend=$Device
        modes=$Modes
        warm_runs=$WarmRuns
        cases=$results
        failures=$failures
    } | ConvertTo-Json -Depth 16 | Set-Content (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Stage91 results: $zip"
}
if ($failures.Count -gt 0) { throw "Stage91 failed with $($failures.Count) issue(s); inspect summary.json" }
