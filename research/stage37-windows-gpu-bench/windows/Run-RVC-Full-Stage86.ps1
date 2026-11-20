param(
    [ValidateSet('cpu','cuda','wgpu')][string]$Device = 'cuda',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(2,16)][int]$WarmRuns = 2,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(1,900)][int]$CoreFrames = 300,
    [ValidateRange(0,1)][double]$IndexRate = 0.75,
    [ValidateSet('baseline','shared','all')][string]$Modes = 'all'
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($p in @($exe,$Voice,$Hubert,$Rmvpe,$InputWav)) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing input: $p" }
}
if ($IndexRate -gt 0 -and -not (Test-Path -LiteralPath $Index -PathType Leaf)) { throw "Missing index: $Index" }
if ($IndexRate -gt 0 -and $Index -eq '-') { throw 'IndexRate>0 requires an index' }
$culture = [Globalization.CultureInfo]::InvariantCulture
$root = Join-Path $PSScriptRoot 'stage86_full_results'
$zip = Join-Path $PSScriptRoot 'RVC_STAGE86_FULL_WAV_RESULTS.zip'
if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
New-Item -ItemType Directory -Path $root -Force | Out-Null
$previous = @{}
foreach ($key in @('RVC_GPU_FAISS','RVC_GPU_F0','RVC_PROFILE','RVC_CONV_WEIGHT_PREPACK','RVC_NSF_CONV_KERNEL','RVC_RMVPE_FUSED_READBACK','RVC_RMVPE_COMPACT_READBACK','RVC_RMVPE_PLANE_READBACK','RVC_GLOBAL_F0','RVC_RMVPE_GRU_PERSISTENT','RVC_NSF_IMPLICIT_GEMM','RVC_HUBERT_FRONTEND_GEMM','RVC_RMVPE_UNET_GEMM','RVC_RMVPE_UNET_CONV2D_GEMM','RVC_RMVPE_UNET_TRANSPOSE2D_GEMM','RVC_RMVPE_UNET_COMPENSATED_TF32','RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32','RVC_DIAGNOSTIC_TRACE','RVC_RMVPE_MEL_REUSE','RVC_NSF_TILE_CORE_FRAMES','RVC_NSF_PAIR_SHARED_TILE','RVC_NSF_FUSED_PAIR_MAX_CHANNELS')) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
}
$allCases = @(
    [pscustomobject]@{ name='baseline_1'; mode='baseline'; shared=$false },
    [pscustomobject]@{ name='shared_1'; mode='shared'; shared=$true },
    [pscustomobject]@{ name='shared_2'; mode='shared'; shared=$true },
    [pscustomobject]@{ name='baseline_2'; mode='baseline'; shared=$false }
)
$cases = @($allCases | Where-Object { $Modes -eq 'all' -or $_.mode -eq $Modes })
if ($cases.Count -eq 0) { throw 'No mode was selected' }
$rows = @()
$failures = @()
try {
    Remove-Item Env:RVC_NSF_PAIR_SHARED_TILE,Env:RVC_NSF_FUSED_PAIR_MAX_CHANNELS,Env:RVC_NSF_TILE_CORE_FRAMES,Env:RVC_PROFILE,Env:RVC_NSF_CONV_KERNEL,Env:RVC_RMVPE_FUSED_READBACK,Env:RVC_RMVPE_COMPACT_READBACK,Env:RVC_RMVPE_PLANE_READBACK,Env:RVC_GLOBAL_F0,Env:RVC_NSF_IMPLICIT_GEMM,Env:RVC_RMVPE_GRU_PERSISTENT,Env:RVC_HUBERT_FRONTEND_GEMM,Env:RVC_RMVPE_UNET_GEMM,Env:RVC_RMVPE_UNET_CONV2D_GEMM,Env:RVC_RMVPE_UNET_TRANSPOSE2D_GEMM,Env:RVC_RMVPE_UNET_COMPENSATED_TF32,Env:RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32,Env:RVC_DIAGNOSTIC_TRACE,Env:RVC_RMVPE_MEL_REUSE -ErrorAction SilentlyContinue
    $env:RVC_CONV_WEIGHT_PREPACK = '1'
    $env:RVC_NSF_IMPLICIT_GEMM = '1'
    $env:RVC_RMVPE_GRU_PERSISTENT = '1'
    if ($Device -eq 'cuda') {
        try { & nvidia-smi -q > (Join-Path $root 'nvidia-smi.txt') 2>&1 }
        catch { $_ | Out-String | Set-Content (Join-Path $root 'nvidia-smi-error.txt') }
    }
    foreach ($case in $cases) {
        try {
        Remove-Item Env:RVC_GPU_FAISS,Env:RVC_GPU_F0,Env:RVC_GLOBAL_F0,Env:RVC_HUBERT_FRONTEND_GEMM,Env:RVC_RMVPE_UNET_GEMM,Env:RVC_RMVPE_UNET_CONV2D_GEMM,Env:RVC_RMVPE_UNET_TRANSPOSE2D_GEMM,Env:RVC_RMVPE_UNET_COMPENSATED_TF32,Env:RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32,Env:RVC_DIAGNOSTIC_TRACE,Env:RVC_RMVPE_MEL_REUSE -ErrorAction SilentlyContinue
        $env:RVC_RMVPE_MEL_REUSE = '1'
        Remove-Item Env:RVC_NSF_TILE_CORE_FRAMES -ErrorAction SilentlyContinue
        if ($case.shared) { $env:RVC_NSF_PAIR_SHARED_TILE = '1' } else { Remove-Item Env:RVC_NSF_PAIR_SHARED_TILE -ErrorAction SilentlyContinue }
        $id = $case.name
        $wav = Join-Path $root "${id}.wav"
        $stdout = Join-Path $root "${id}.stdout.log"
        $stderr = Join-Path $root "${id}.stderr.log"
        $argsArray = @('infer-wav-full-benchmark',$Voice,$Hubert,$Rmvpe,$Index,$InputWav,$wav,
            $IndexRate.ToString($culture),$Device,[string]$WarmRuns,[string]$CoreFrames,[string]$TensorBatchSize)
        $quoted = @($argsArray | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
        Write-Host "Stage86 full WAV ${id}: device=$Device, repeats=$WarmRuns, chunk=$CoreFrames frames, batch=$TensorBatchSize, nsfGemm=true, persistentGRU=true, nsf_shared_pair_tile=$($case.shared)"
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = $exe
        $psi.Arguments = ($quoted -join ' ')
        $psi.WorkingDirectory = $PSScriptRoot
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $psi
        try {
            if (-not $process.Start()) { throw "Stage86 ${id} process did not start" }
            $outputTask = $process.StandardOutput.ReadToEndAsync()
            $errorTask = $process.StandardError.ReadToEndAsync()
            $process.WaitForExit()
            [IO.File]::WriteAllText($stdout, $outputTask.GetAwaiter().GetResult(), [Text.Encoding]::UTF8)
            [IO.File]::WriteAllText($stderr, $errorTask.GetAwaiter().GetResult(), [Text.Encoding]::UTF8)
            $exitCode = $process.ExitCode
        } finally {
            $process.Dispose()
        }
        if ($exitCode -ne 0) {
            Get-Content -LiteralPath $stderr -Tail 35 -ErrorAction SilentlyContinue
            throw "Stage86 ${id} failed (exit $exitCode)"
        }
        $report = Get-Content -LiteralPath "${wav}.benchmark.json" -Raw | ConvertFrom-Json
        if ($report.stage -ne 63 -or $report.graph_replay -ne $false -or $report.backend -ne $Device -or $report.inference_ms.Count -ne $WarmRuns -or $report.model_input_seconds -lt 10 -or [bool]$report.global_f0 -ne $false -or $report.nsf_execution -ne 'sequential' -or $report.nsf_conv_execution -ne 'implicit_gemm' -or $report.rmvpe_gru_execution -ne 'persistent' -or $report.hubert_frontend_conv_execution -ne 'direct' -or $report.rmvpe_unet_conv_execution -ne 'direct' -or [bool]$report.rmvpe_mel_reuse -ne $true -or [bool]$report.nsf_pair_shared_tile -ne [bool]$case.shared -or $null -ne $report.nsf_fused_pair_max_channels -or $null -ne $report.nsf_tile_core_frames ) {
            throw "Stage86 ${id}: invalid report or source WAV is shorter than 10 seconds"
        }
        $expected = [int64]$report.output_samples
        $wavBytes = [IO.File]::ReadAllBytes($wav)
        $position = 12
        $pcm = $null
        while ($position + 8 -le $wavBytes.Length) {
            $chunk = [Text.Encoding]::ASCII.GetString($wavBytes,$position,4)
            $size = [BitConverter]::ToInt32($wavBytes,$position+4)
            if ($size -lt 0 -or $position + 8 + $size -gt $wavBytes.Length) { throw "Invalid output WAV chunk: $wav" }
            if ($chunk -eq 'data') {
                if (($size % 2) -ne 0) { throw "Not PCM16 aligned: $wav" }
                $pcm = [int16[]]::new($size / 2)
                [Buffer]::BlockCopy($wavBytes,$position+8,$pcm,0,$size)
            }
            $position += 8 + $size + ($size % 2)
        }
        if ($null -eq $pcm -or $pcm.Length -ne $expected) { throw "Stage86 ${id}: output PCM size mismatch" }
        $record = [pscustomobject]@{
            mode = $id
            nsf_pair_shared_tile = $report.nsf_pair_shared_tile
            nsf_conv_execution = $report.nsf_conv_execution
            rmvpe_gru_execution = $report.rmvpe_gru_execution
            hubert_frontend_conv_execution = $report.hubert_frontend_conv_execution
            rmvpe_unet_conv_execution = $report.rmvpe_unet_conv_execution
            backend = $Device
            source_seconds = $report.source_seconds
            output_samples = $report.output_samples
            duration_seconds = $report.output_samples / 40000.0
            load_ms = $report.load_ms
            first_inference_ms = $report.first_inference_ms
            warm_median_ms = $report.warm_median_ms
            warm_rtf = $report.warm_rtf
            prepare_ms = $report.prepare_ms
            postprocess_write_ms = $report.postprocess_write_ms
            iteration_exact = $report.iterations_exact_f32
            max_abs_iteration_delta = $report.max_abs_iteration_delta
            chunk_counts = $report.chunk_counts
            inference_ms = $report.inference_ms
            sha256 = (Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash
        }
        $rows += $record
        Write-Host ('  full={0:N2}s, load={1:N0}ms, first={2:N1}ms, warm={3:N1}ms, RTF={4:N3}, chunks={5}, stable={6}' -f $record.source_seconds,$record.load_ms,$record.first_inference_ms,$record.warm_median_ms,$record.warm_rtf,$record.chunk_counts[0],$record.iteration_exact)
        } catch {
            $failure = "Stage86 $($case.name): $($_.Exception.Message)"
            $failures += [pscustomobject]@{ mode=$case.name; error=$failure }
            Write-Warning $failure
            $failure | Set-Content -LiteralPath (Join-Path $root "$($case.name).error.txt") -Encoding UTF8
        }
    }
    $pairwise = @()
    if (@($rows | Where-Object { $_.mode -eq 'baseline_1' }).Count -eq 1) {
        $base = Join-Path $root 'baseline_1.wav'
        foreach ($other in @($rows | Where-Object { $_.mode -ne 'baseline_1' })) {
            $a = [IO.File]::ReadAllBytes($base)
            $b = [IO.File]::ReadAllBytes((Join-Path $root ($other.mode + '.wav')))
            if ($a.Length -ne $b.Length) { throw "Stage86 output byte count differs: $($other.mode)" }
            $different = 0
            $maxAbs = 0
            $squared = 0.0
            $sampleCount = [int64]$other.output_samples
            $sourcePcm = [int16[]]::new($sampleCount)
            $otherPcm = [int16[]]::new($sampleCount)
                    $dataStart = 44
            if ($a.Length -ne $dataStart + $sampleCount*2) { throw 'Unexpected WAV data layout for PCM comparison' }
            [Buffer]::BlockCopy($a,$dataStart,$sourcePcm,0,$sampleCount*2)
            [Buffer]::BlockCopy($b,$dataStart,$otherPcm,0,$sampleCount*2)
            for ($i=0; $i -lt $sampleCount; $i++) {
                $delta = [int]$sourcePcm[$i] - [int]$otherPcm[$i]
                if ($delta -ne 0) { $different++ }
                $maxAbs = [Math]::Max($maxAbs,[Math]::Abs($delta))
                $squared += [double]$delta*$delta
            }
            $pairwise += [pscustomobject]@{mode=$other.mode; unequal_samples=$different; sample_count=$sampleCount; max_abs_pcm16=$maxAbs; rmse_pcm16=[Math]::Sqrt($squared/$sampleCount)}
            Write-Host ('  baseline vs {0}: max_abs={1}, unequal={2}/{3}, RMSE={4:N5}' -f $other.mode,$maxAbs,$different,$sampleCount,[Math]::Sqrt($squared/$sampleCount))
            if ($other.mode -eq 'baseline_2' -and $different -ne 0) { $failures += [pscustomobject]@{mode=$other.mode; error="Baseline repeats produced different output: $different samples"} }
        }
    }
    if ($Modes -eq 'all' -and $rows.Count -ne 4) {
        $failures += [pscustomobject]@{mode='all';error='Incomplete ABBA comparison; four successful cases required'}
    }
    $summary = [pscustomobject]@{
        stage = 86
        experiment = 'nsf-one-launch-temporal-shared-pair-vs-implicit-gemm'
        backend = $Device
        entire_source = $true
        production_residual_execution = 'sequential'
        graph_replay = $false
        warm_runs = $WarmRuns
        modes = $Modes
        stable_pcm_required = $false
        shared_tile_time = 32
        cases = $rows
        parity = $pairwise
        failures = $failures
    }
    $summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Stage86 results: $zip"
}

if ($failures.Count -gt 0) { throw "Stage86 failed parity or execution; inspect RVC_STAGE86_FULL_WAV_RESULTS.zip" }
