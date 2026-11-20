param(
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(2, 8)][int]$Runs = 2,
    [ValidateRange(1, 900)][int]$CoreFrames = 300,
    [ValidateRange(1, 32)][int]$TensorBatchSize = 2,
    [ValidateRange(0, 1)][double]$IndexRate = 0.75,
    [ValidateSet('all', 'conv2d', 'transpose2d')][string]$Modes = 'all'
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($item in @($exe, $Voice, $Hubert, $Rmvpe, $Index, $InputWav)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw "Missing file: $item" }
}
$exe = (Resolve-Path -LiteralPath $exe).Path
$Voice = (Resolve-Path -LiteralPath $Voice).Path
$Hubert = (Resolve-Path -LiteralPath $Hubert).Path
$Rmvpe = (Resolve-Path -LiteralPath $Rmvpe).Path
$Index = (Resolve-Path -LiteralPath $Index).Path
$InputWav = (Resolve-Path -LiteralPath $InputWav).Path
$root = Join-Path $PSScriptRoot 'stage75_exact_f0_results'
$archive = Join-Path $PSScriptRoot 'RVC_STAGE75_EXACT_F0_RESULTS.zip'
if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
New-Item -ItemType Directory -Path $root -Force | Out-Null
$keys = @('RVC_F0_AUDIT_DIR','RVC_CONV_WEIGHT_PREPACK','RVC_NSF_IMPLICIT_GEMM','RVC_NSF_CONV1D_GEMM',
    'RVC_NSF_TRANSPOSE_GEMM','RVC_NSF_CONV_KERNEL','RVC_RMVPE_GRU_PERSISTENT','RVC_RMVPE_UNET_GEMM',
    'RVC_RMVPE_UNET_CONV2D_GEMM','RVC_RMVPE_UNET_TRANSPOSE2D_GEMM','RVC_HUBERT_FRONTEND_GEMM',
    'RVC_GPU_FAISS','RVC_GPU_F0','RVC_GLOBAL_F0','RVC_PROFILE','RVC_DIAGNOSTIC_TRACE')
$previous = @{}
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
$allCases = @(
    [pscustomobject]@{name='direct'; conv=$false; transpose=$false},
    [pscustomobject]@{name='conv2d'; conv=$true; transpose=$false},
    [pscustomobject]@{name='transpose2d'; conv=$false; transpose=$true},
    [pscustomobject]@{name='both'; conv=$true; transpose=$true}
)
$cases = @($allCases | Where-Object {
    $Modes -eq 'all' -or $_.name -eq 'direct' -or $_.name -eq $Modes
})
$rows = @()
$comparisons = @()
$failures = @()
$culture = [Globalization.CultureInfo]::InvariantCulture
try {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $null, 'Process') }
    $env:RVC_CONV_WEIGHT_PREPACK = '1'
    $env:RVC_NSF_IMPLICIT_GEMM = '1'
    $env:RVC_RMVPE_GRU_PERSISTENT = '1'
    try { & nvidia-smi -q > (Join-Path $root 'nvidia-smi.txt') 2>&1 } catch {}
    foreach ($case in $cases) {
        $name = $case.name
        try {
            Remove-Item Env:RVC_RMVPE_UNET_CONV2D_GEMM, Env:RVC_RMVPE_UNET_TRANSPOSE2D_GEMM -ErrorAction SilentlyContinue
            if ($case.conv) { $env:RVC_RMVPE_UNET_CONV2D_GEMM = '1' }
            if ($case.transpose) { $env:RVC_RMVPE_UNET_TRANSPOSE2D_GEMM = '1' }
            $dir = Join-Path $root $name
            $f0Dir = Join-Path $dir 'f0_chunks'
            New-Item -ItemType Directory -Path $f0Dir -Force | Out-Null
            $env:RVC_F0_AUDIT_DIR = $f0Dir
            $wav = Join-Path $dir 'converted.wav'
            $stdout = Join-Path $dir 'stdout.log'
            $stderr = Join-Path $dir 'stderr.log'
            Write-Host "Stage75 exact full-RVC F0 audit ${name}: Conv2d=$($case.conv) Transpose2d=$($case.transpose)"
            $argv = @('infer-wav-full-benchmark', $Voice, $Hubert, $Rmvpe, $Index, $InputWav,
                $wav, $IndexRate.ToString($culture), 'cuda', [string]$Runs, [string]$CoreFrames, [string]$TensorBatchSize)
            $psi = [System.Diagnostics.ProcessStartInfo]::new()
            $psi.FileName = $exe
            $psi.WorkingDirectory = $PSScriptRoot
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.Arguments = (@($argv | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }) -join ' ')
            $proc = [System.Diagnostics.Process]::new()
            $proc.StartInfo = $psi
            try {
                if (-not $proc.Start()) { throw "Cannot start $name" }
                $outTask = $proc.StandardOutput.ReadToEndAsync()
                $errTask = $proc.StandardError.ReadToEndAsync()
                $proc.WaitForExit()
                [IO.File]::WriteAllText($stdout, $outTask.GetAwaiter().GetResult(), [Text.Encoding]::UTF8)
                [IO.File]::WriteAllText($stderr, $errTask.GetAwaiter().GetResult(), [Text.Encoding]::UTF8)
                if ($proc.ExitCode -ne 0) { throw "$name returned exit code $($proc.ExitCode)" }
            } finally { $proc.Dispose() }
            $report = Get-Content -LiteralPath "$wav.benchmark.json" -Raw | ConvertFrom-Json
            $expected = if ($case.conv -and $case.transpose) {'implicit_gemm'} elseif ($case.conv) {'conv2d_implicit_gemm'} elseif ($case.transpose) {'transpose2d_gemm'} else {'direct'}
            if ($report.backend -ne 'cuda' -or $report.rmvpe_unet_conv_execution -ne $expected -or
                $report.nsf_conv_execution -ne 'implicit_gemm' -or $report.rmvpe_gru_execution -ne 'persistent' -or
                $report.graph_replay -ne $false -or $report.iterations_exact_f32 -ne $true -or
                $report.inference_ms.Count -ne $Runs) {
                throw "$name benchmark metadata or deterministic result invalid"
            }
            $f0Meta = @(Get-ChildItem -LiteralPath $f0Dir -Filter 'chunk_*.json' -File)
            if ($f0Meta.Count -ne $report.chunk_counts[0]) { throw "$name F0 audit incomplete: $($f0Meta.Count) / $($report.chunk_counts[0])" }
            $rows += [pscustomobject]@{
                name=$name; conv2d_gemm=$case.conv; transpose2d_gemm=$case.transpose;
                warm_ms=$report.warm_median_ms; input_seconds=$report.model_input_seconds;
                output_samples=$report.output_samples; f0_chunks=$f0Meta.Count;
                wav_sha256=(Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash
            }
            Write-Host ('  full-WAV warm={0:N2} ms; audited F0 chunks={1}' -f $report.warm_median_ms, $f0Meta.Count)
        } catch {
            $failures += [pscustomobject]@{mode=$name; error=$_.Exception.Message}
            Write-Warning "Stage75 ${name}: $($_.Exception.Message)"
            $_.Exception.ToString() | Set-Content -LiteralPath (Join-Path $root "$name.error.txt")
        } finally { Remove-Item Env:RVC_F0_AUDIT_DIR -ErrorAction SilentlyContinue }
    }
    if (@($rows | Where-Object name -eq 'direct').Count -eq 1) {
        $referenceDir = Join-Path $root 'direct/f0_chunks'
        foreach ($row in @($rows | Where-Object name -ne 'direct')) {
            $mode = $row.name
            $differentPitch = 0
            $differentVoicing = 0
            $differentF0 = 0
            $totalFrames = 0
            $maxF0 = 0.0
            $squaredF0 = 0.0
            $worstChunk = -1
            $perChunk = @()
            foreach ($referenceF0 in @(Get-ChildItem -LiteralPath $referenceDir -Filter 'chunk_*.f0.f32le' -File | Sort-Object Name)) {
                $f0Name = $referenceF0.Name
                $candidatePath = Join-Path (Join-Path $root "$mode/f0_chunks") $f0Name
                $refBytes = [IO.File]::ReadAllBytes($referenceF0.FullName)
                $newBytes = [IO.File]::ReadAllBytes($candidatePath)
                if ($refBytes.Length -ne $newBytes.Length -or ($refBytes.Length % 4) -ne 0) { throw "F0 length mismatch for $mode / $f0Name" }
                $count = [int]($refBytes.Length / 4)
                $a = [float[]]::new($count)
                $b = [float[]]::new($count)
                [Buffer]::BlockCopy($refBytes, 0, $a, 0, $refBytes.Length)
                [Buffer]::BlockCopy($newBytes, 0, $b, 0, $newBytes.Length)
                $pitchFile = $referenceF0.Name.Replace('.f0.f32le', '.pitch.u8')
                $p0 = [IO.File]::ReadAllBytes((Join-Path $referenceDir $pitchFile))
                $p1 = [IO.File]::ReadAllBytes((Join-Path (Join-Path $root "$mode/f0_chunks") $pitchFile))
                if ($p0.Length -ne $count -or $p1.Length -ne $count) { throw "pitch count mismatch for $mode / $pitchFile" }
                $chunkMax = 0.0; $chunkCoarse = 0; $chunkVoiced = 0
                for ($i = 0; $i -lt $count; $i++) {
                    if ($a[$i] -ne $b[$i]) { $differentF0++ }
                    if ($p0[$i] -ne $p1[$i]) { $differentPitch++; $chunkCoarse++ }
                    if (($a[$i] -gt 0) -ne ($b[$i] -gt 0)) { $differentVoicing++; $chunkVoiced++ }
                    $delta = [Math]::Abs([double]$a[$i] - [double]$b[$i])
                    $chunkMax = [Math]::Max($chunkMax, $delta)
                    $squaredF0 += $delta * $delta
                }
                if ($chunkMax -gt $maxF0) { $maxF0 = $chunkMax; $worstChunk = $f0Name }
                $totalFrames += $count
                $perChunk += [pscustomobject]@{chunk=$f0Name; frames=$count; max_abs_hz=$chunkMax; coarse_changes=$chunkCoarse; voiced_changes=$chunkVoiced}
            }
            $perChunk | Export-Csv -LiteralPath (Join-Path $root "$mode.f0-chunks.csv") -NoTypeInformation -Encoding UTF8
            $comparisons += [pscustomobject]@{
                mode=$mode; frames=$totalFrames; different_f0=$differentF0;
                different_pitch_bins=$differentPitch; different_voiced=$differentVoicing;
                max_f0_error_hz=$maxF0; rmse_f0_hz=[Math]::Sqrt($squaredF0 / $totalFrames);
                worst_chunk=$worstChunk
            }
        }
    }
    [pscustomobject]@{
        stage=75; experiment='exact production-chunk RMVPE F0/pitch audit';
        graph_replay=$false; cases=$rows; f0_parity=$comparisons; failures=$failures;
        notes='All input WAV resampling, high-pass filtering, reflect halos, batch-size and alignment use real RVC code. F0 data captured after pitch shifting immediately before TextEncoder and NSF.'
    } | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $archive -CompressionLevel Optimal
    Write-Host "Stage75 exact F0 diagnostics: $archive"
}
