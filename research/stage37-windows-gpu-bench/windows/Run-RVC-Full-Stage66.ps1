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
    [ValidateSet('both','legacy','shared')][string]$F0Modes = 'both'
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($p in @($exe,$Voice,$Hubert,$Rmvpe,$InputWav)) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing input: $p" }
}
if ($IndexRate -gt 0 -and -not (Test-Path -LiteralPath $Index -PathType Leaf)) { throw "Missing index: $Index" }
if ($IndexRate -gt 0 -and $Index -eq '-') { throw 'IndexRate>0 requires an index' }
$culture = [Globalization.CultureInfo]::InvariantCulture
$root = Join-Path $PSScriptRoot 'stage66_full_results'
$zip = Join-Path $PSScriptRoot 'RVC_STAGE66_FULL_WAV_RESULTS.zip'
if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
New-Item -ItemType Directory -Path $root -Force | Out-Null
$previous = @{}
foreach ($key in @('RVC_GPU_FAISS','RVC_GPU_F0','RVC_PROFILE','RVC_CONV_WEIGHT_PREPACK','RVC_NSF_CONV_KERNEL','RVC_RMVPE_FUSED_READBACK','RVC_RMVPE_COMPACT_READBACK','RVC_RMVPE_PLANE_READBACK','RVC_GLOBAL_F0')) {
    $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
}
$allCases = @(
    [pscustomobject]@{ name='legacy'; sharedF0=$false },
    [pscustomobject]@{ name='shared'; sharedF0=$true }
)
$cases = @($allCases | Where-Object { $F0Modes -eq 'both' -or $_.name -eq $F0Modes })
if ($cases.Count -eq 0) { throw 'No mode was selected' }
$rows = @()
try {
    Remove-Item Env:RVC_PROFILE,Env:RVC_NSF_CONV_KERNEL,Env:RVC_RMVPE_FUSED_READBACK,Env:RVC_RMVPE_COMPACT_READBACK,Env:RVC_RMVPE_PLANE_READBACK,Env:RVC_GLOBAL_F0 -ErrorAction SilentlyContinue
    $env:RVC_CONV_WEIGHT_PREPACK = '1'
    if ($Device -eq 'cuda') {
        try { & nvidia-smi -q > (Join-Path $root 'nvidia-smi.txt') 2>&1 }
        catch { $_ | Out-String | Set-Content (Join-Path $root 'nvidia-smi-error.txt') }
    }
    foreach ($case in $cases) {
        Remove-Item Env:RVC_GPU_FAISS,Env:RVC_GPU_F0,Env:RVC_GLOBAL_F0 -ErrorAction SilentlyContinue
        if ($case.sharedF0) { $env:RVC_GLOBAL_F0 = '1' }
        $id = $case.name
        $wav = Join-Path $root "${id}.wav"
        $stdout = Join-Path $root "${id}.stdout.log"
        $stderr = Join-Path $root "${id}.stderr.log"
        $argsArray = @('infer-wav-full-benchmark',$Voice,$Hubert,$Rmvpe,$Index,$InputWav,$wav,
            $IndexRate.ToString($culture),$Device,[string]$WarmRuns,[string]$CoreFrames,[string]$TensorBatchSize)
        $quoted = @($argsArray | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
        Write-Host "Stage66 full WAV ${id}: device=$Device, repeats=$WarmRuns, chunk=$CoreFrames frames, batch=$TensorBatchSize, sharedF0=$($case.sharedF0)"
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
            if (-not $process.Start()) { throw "Stage66 ${id} process did not start" }
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
            throw "Stage66 ${id} failed (exit $exitCode)"
        }
        $report = Get-Content -LiteralPath "${wav}.benchmark.json" -Raw | ConvertFrom-Json
        if ($report.stage -ne 63 -or $report.graph_replay -ne $false -or $report.backend -ne $Device -or $report.inference_ms.Count -ne $WarmRuns -or $report.model_input_seconds -lt 10 -or [bool]$report.global_f0 -ne $case.sharedF0 -or $report.nsf_execution -ne 'sequential' ) {
            throw "Stage66 ${id}: invalid report or source WAV is shorter than 10 seconds"
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
        if ($null -eq $pcm -or $pcm.Length -ne $expected) { throw "Stage66 ${id}: output PCM size mismatch" }
        $record = [pscustomobject]@{
            mode = $id
            global_f0 = $case.sharedF0
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
    }
    $pairwise = @()
    if (@($rows | Where-Object { $_.mode -eq 'legacy' }).Count -eq 1) {
        $base = Join-Path $root 'legacy.wav'
        foreach ($other in @($rows | Where-Object { $_.mode -ne 'legacy' })) {
            $a = [IO.File]::ReadAllBytes($base)
            $b = [IO.File]::ReadAllBytes((Join-Path $root ($other.mode + '.wav')))
            if ($a.Length -ne $b.Length) { throw "Stage66 output byte count differs: $($other.mode)" }
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
            Write-Host ('  legacy vs {0}: max_abs={1}, unequal={2}/{3}, RMSE={4:N5}' -f $other.mode,$maxAbs,$different,$sampleCount,[Math]::Sqrt($squared/$sampleCount))
        }
    }
    $summary = [pscustomobject]@{
        stage = 66
        experiment = 'full-wav-shared-f0-ab'
        backend = $Device
        entire_source = $true
        production_residual_execution = 'sequential'
        graph_replay = $false
        warm_runs = $WarmRuns
        f0_modes = $F0Modes
        cases = $rows
        parity = $pairwise
    }
    $summary | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Stage66 results: $zip"
}
