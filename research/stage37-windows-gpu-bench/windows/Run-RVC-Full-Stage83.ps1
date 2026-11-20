param(
    [ValidateSet('cuda')][string]$Device = 'cuda',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(2,16)][int]$WarmRuns = 2,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(1,900)][int]$CoreFrames = 300,
    [ValidateRange(0,1)][double]$IndexRate = 0.75,
    [ValidateSet('baseline','online','flash','hybrid','all')][string]$Modes = 'all'
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($path in @($exe,$Voice,$Hubert,$Rmvpe,$InputWav)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing: $path" }
}
if ($IndexRate -gt 0 -and -not (Test-Path -LiteralPath $Index -PathType Leaf)) { throw "Missing index: $Index" }
$culture = [Globalization.CultureInfo]::InvariantCulture
$root = Join-Path $PSScriptRoot 'stage83_full_results'
$zip = Join-Path $PSScriptRoot 'RVC_STAGE83_FULL_WAV_RESULTS.zip'
if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Force -Recurse }
New-Item -Path $root -ItemType Directory -Force | Out-Null
$keys = @('RVC_ATTENTION_ONLINE','RVC_HUBERT_ATTENTION_FLASH','RVC_GPU_FAISS','RVC_GPU_F0','RVC_PROFILE','RVC_GLOBAL_F0','RVC_RMVPE_MEL_REUSE','RVC_RMVPE_ENCODER0_REUSE','RVC_NSF_IMPLICIT_GEMM','RVC_NSF_CONV_KERNEL','RVC_RMVPE_GRU_PERSISTENT','RVC_HUBERT_FRONTEND_GEMM','RVC_RMVPE_UNET_GEMM','RVC_RMVPE_UNET_CONV2D_GEMM','RVC_RMVPE_UNET_TRANSPOSE2D_GEMM','RVC_RMVPE_UNET_COMPENSATED_TF32','RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32','RVC_DIAGNOSTIC_TRACE','RVC_CONV_WEIGHT_PREPACK')
$prior = @{}
foreach ($key in $keys) { $prior[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$cases = if ($Modes -eq 'all') { @('baseline','online','flash','hybrid','hybrid','flash','online','baseline') }
    elseif ($Modes -eq 'baseline') { @('baseline','baseline') }
    else { @('baseline',$Modes,$Modes,'baseline') }
$rows = @()
$failures = @()
$visits = @{}
try {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    $env:RVC_CONV_WEIGHT_PREPACK = '1'
    $env:RVC_NSF_IMPLICIT_GEMM = '1'
    $env:RVC_RMVPE_GRU_PERSISTENT = '1'
    try { & nvidia-smi -q > (Join-Path $root 'nvidia-smi.txt') 2>&1 } catch { $_ | Out-String | Set-Content (Join-Path $root 'nvidia-smi-error.txt') }
    foreach ($mode in $cases) {
        if (-not $visits.ContainsKey($mode)) { $visits[$mode] = 0 }
        $visits[$mode]++
        $id = "${mode}_$($visits[$mode])"
        $env:RVC_ATTENTION_ONLINE = $null
        $env:RVC_HUBERT_ATTENTION_FLASH = $null
        if ($mode -in @('online','hybrid')) { $env:RVC_ATTENTION_ONLINE = '1' }
        if ($mode -in @('flash','hybrid')) { $env:RVC_HUBERT_ATTENTION_FLASH = '1' }
        $wav = Join-Path $root "${id}.wav"
        $stdout = Join-Path $root "${id}.stdout.log"
        $stderr = Join-Path $root "${id}.stderr.log"
        $argsArray = @('infer-wav-full-benchmark',$Voice,$Hubert,$Rmvpe,$Index,$InputWav,$wav,$IndexRate.ToString($culture),$Device,[string]$WarmRuns,[string]$CoreFrames,[string]$TensorBatchSize)
        $quoted = @($argsArray | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
        Write-Host "Stage83 $id, core=$CoreFrames, batch=$TensorBatchSize, warm runs=$WarmRuns"
        try {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = $exe
            $psi.Arguments = $quoted -join ' '
            $psi.WorkingDirectory = $PSScriptRoot
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $process = New-Object System.Diagnostics.Process
            $process.StartInfo = $psi
            try {
                if (-not $process.Start()) { throw 'Process failed to start' }
                $outTask = $process.StandardOutput.ReadToEndAsync()
                $errTask = $process.StandardError.ReadToEndAsync()
                $process.WaitForExit()
                [IO.File]::WriteAllText($stdout,$outTask.GetAwaiter().GetResult(),[Text.Encoding]::UTF8)
                [IO.File]::WriteAllText($stderr,$errTask.GetAwaiter().GetResult(),[Text.Encoding]::UTF8)
                $exitCode = $process.ExitCode
            } finally { $process.Dispose() }
            if ($exitCode -ne 0) { throw "Inference exit code $exitCode" }
            $report = Get-Content -LiteralPath "${wav}.benchmark.json" -Raw | ConvertFrom-Json
            $hubertExpected = if ($mode -in @('flash','hybrid')) { 'hubert_flash_unit' } elseif ($mode -eq 'online') { 'streaming_online_softmax' } else { 'dense' }
            $textExpected = if ($mode -in @('online','hybrid')) { 'streaming_relative_online_softmax' } else { 'dense_relative' }
            if ($report.stage -ne 63 -or $report.graph_replay -ne $false -or $report.backend -ne 'cuda' -or $report.inference_ms.Count -ne $WarmRuns -or $report.attention_execution -ne $hubertExpected -or $report.text_encoder_attention_execution -ne $textExpected) {
                throw "Unexpected stage, device, attention mode or number of runs: $id"
            }
            $bytes = [IO.File]::ReadAllBytes($wav)
            $position = 12
            $pcmStart = -1
            $pcmBytes = -1
            while ($position + 8 -le $bytes.Length) {
                $tag = [Text.Encoding]::ASCII.GetString($bytes,$position,4)
                $size = [BitConverter]::ToInt32($bytes,$position+4)
                if ($size -lt 0 -or $position + 8 + $size -gt $bytes.Length) { throw "Malformed WAV: $id" }
                if ($tag -eq 'data') { $pcmStart = $position+8; $pcmBytes = $size; break }
                $position += 8 + $size + ($size % 2)
            }
            if ($pcmBytes -ne ([int64]$report.output_samples * 2) -or $pcmStart -lt 0) { throw "PCM size mismatch: $id" }
            $rows += [pscustomobject]@{
                id=$id; mode=$mode; warm_median_ms=$report.warm_median_ms;
                warm_rtf=$report.warm_rtf; first_inference_ms=$report.first_inference_ms;
                inference_ms=$report.inference_ms; iteration_exact=$report.iterations_exact_f32;
                output_samples=$report.output_samples; chunk_counts=$report.chunk_counts;
                attention_execution=$report.attention_execution;
                text_encoder_attention_execution=$report.text_encoder_attention_execution;
                wav_sha256=(Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash
            }
        } catch {
            $failures += [pscustomobject]@{id=$id; error=$_.Exception.Message}
            Write-Warning "Stage83 $id failed: $($_.Exception.Message)"
        }
    }
    $pairs = @()
    $baseline = @($rows | Where-Object { $_.mode -eq 'baseline' } | Select-Object -First 1)
    if ($baseline.Count -gt 0) {
        $baseBytes = [IO.File]::ReadAllBytes((Join-Path $root "$($baseline[0].id).wav"))
        $count = [int64]$baseline[0].output_samples
        foreach ($case in @($rows | Where-Object { $_.mode -ne 'baseline' })) {
            $candidateBytes = [IO.File]::ReadAllBytes((Join-Path $root "$($case.id).wav"))
            if ($baseBytes.Length -ne $candidateBytes.Length) { $failures += [pscustomobject]@{id=$case.id;error='WAV length mismatch'}; continue }
            $a = [int16[]]::new($count)
            $b = [int16[]]::new($count)
            [Buffer]::BlockCopy($baseBytes,44,$a,0,$count*2)
            [Buffer]::BlockCopy($candidateBytes,44,$b,0,$count*2)
            $unequal=0; $maxAbs=0; $sse=0.0
            for ($i=0;$i -lt $count;$i++) {
                $d=[int]$a[$i]-[int]$b[$i]
                if ($d -ne 0) { $unequal++ }
                $maxAbs=[Math]::Max($maxAbs,[Math]::Abs($d))
                $sse += [double]$d * $d
            }
            $pairs += [pscustomobject]@{id=$case.id;unequal_samples=$unequal;max_abs_pcm16=$maxAbs;rmse_pcm16=[Math]::Sqrt($sse/$count)}
        }
    }
    $summary = [pscustomobject]@{
        stage=83; experiment='dense-vs-online-vs-cubek-flash-attention';
        backend='cuda'; graph_replay=$false; warm_runs=$WarmRuns;
        core_frames=$CoreFrames; batch_size=$TensorBatchSize;
        measurement_order=$cases; cases=$rows; parity=$pairs; failures=$failures
    }
    $summary | ConvertTo-Json -Depth 20 | Set-Content (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $prior.Keys) { [Environment]::SetEnvironmentVariable($key,$prior[$key],'Process') }
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Stage83 results: $zip"
}
if ($failures.Count -gt 0) { throw 'Stage83 encountered errors; inspect results ZIP' }
