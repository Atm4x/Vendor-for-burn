param(
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateSet('all','nsf','unet')][string]$Suite = 'all',
    [ValidateRange(2,8)][int]$WarmRuns = 2,
    [ValidateRange(1,900)][int]$CoreFrames = 300,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(0,1)][double]$IndexRate = 0.75
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($file in @($exe,$Voice,$Hubert,$Rmvpe,$Index,$InputWav)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing input: $file" }
}
$exe = (Resolve-Path -LiteralPath $exe).Path
$Voice = (Resolve-Path -LiteralPath $Voice).Path
$Hubert = (Resolve-Path -LiteralPath $Hubert).Path
$Rmvpe = (Resolve-Path -LiteralPath $Rmvpe).Path
$Index = (Resolve-Path -LiteralPath $Index).Path
$InputWav = (Resolve-Path -LiteralPath $InputWav).Path
$root = Join-Path $PSScriptRoot 'stage73_full_results'
$zip = Join-Path $PSScriptRoot 'RVC_STAGE73_FULL_WAV_RESULTS.zip'
if (Test-Path $root) { Remove-Item $root -Recurse -Force }
New-Item -ItemType Directory -Force -Path $root | Out-Null
$keys = @(
    'RVC_GPU_FAISS','RVC_GPU_F0','RVC_PROFILE','RVC_DIAGNOSTIC_TRACE',
    'RVC_CONV_WEIGHT_PREPACK','RVC_NSF_CONV_KERNEL','RVC_NSF_IMPLICIT_GEMM',
    'RVC_RMVPE_GRU_PERSISTENT','RVC_RMVPE_UNET_GEMM',
    'RVC_RMVPE_UNET_CONV2D_GEMM','RVC_RMVPE_UNET_TRANSPOSE2D_GEMM',
    'RVC_GLOBAL_F0','RVC_HUBERT_FRONTEND_GEMM',
    'RVC_RMVPE_FUSED_READBACK','RVC_RMVPE_COMPACT_READBACK','RVC_RMVPE_PLANE_READBACK'
)
$previous = @{}
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$cases = @(
    [pscustomobject]@{name='baseline'; group='baseline'; nsf='gemm'; hubert=$false; conv=$false; transpose=$false},
    [pscustomobject]@{name='hubert'; group='unet'; nsf='gemm'; hubert=$true; conv=$false; transpose=$false},
    [pscustomobject]@{name='unet_conv2d'; group='unet'; nsf='gemm'; hubert=$false; conv=$true; transpose=$false},
    [pscustomobject]@{name='unet_transpose'; group='unet'; nsf='gemm'; hubert=$false; conv=$false; transpose=$true},
    [pscustomobject]@{name='unet_both'; group='unet'; nsf='gemm'; hubert=$false; conv=$true; transpose=$true},
    [pscustomobject]@{name='nsf_prepacked'; group='nsf'; nsf='prepacked'; hubert=$false; conv=$false; transpose=$false},
    [pscustomobject]@{name='nsf_quad'; group='nsf'; nsf='quad'; hubert=$false; conv=$false; transpose=$false},
    [pscustomobject]@{name='nsf_quad_interior'; group='nsf'; nsf='quad_interior'; hubert=$false; conv=$false; transpose=$false}
) | Where-Object { $_.group -eq 'baseline' -or $Suite -eq 'all' -or $_.group -eq $Suite }
$rows = @()
$errors = @()
$culture = [Globalization.CultureInfo]::InvariantCulture
try {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    $env:RVC_CONV_WEIGHT_PREPACK = '1'
    $env:RVC_RMVPE_GRU_PERSISTENT = '1'
    try { & nvidia-smi -q > (Join-Path $root 'nvidia-smi.txt') 2>&1 } catch {}
    foreach ($case in $cases) {
        $name = $case.name
        try {
            Remove-Item Env:RVC_NSF_CONV_KERNEL,Env:RVC_NSF_IMPLICIT_GEMM,Env:RVC_HUBERT_FRONTEND_GEMM,Env:RVC_RMVPE_UNET_CONV2D_GEMM,Env:RVC_RMVPE_UNET_TRANSPOSE2D_GEMM -ErrorAction SilentlyContinue
            if ($case.nsf -eq 'gemm') { $env:RVC_NSF_IMPLICIT_GEMM = '1' }
            if ($case.nsf -in @('quad','quad_interior')) { $env:RVC_NSF_CONV_KERNEL = $case.nsf }
            if ($case.hubert) { $env:RVC_HUBERT_FRONTEND_GEMM = '1' }
            if ($case.conv) { $env:RVC_RMVPE_UNET_CONV2D_GEMM = '1' }
            if ($case.transpose) { $env:RVC_RMVPE_UNET_TRANSPOSE2D_GEMM = '1' }
            $wav = Join-Path $root "$name.wav"
            $stdout = Join-Path $root "$name.stdout.log"
            $stderr = Join-Path $root "$name.stderr.log"
            Write-Host "Stage73 full WAV: $name NSF=$($case.nsf) HuBERT_GEMM=$($case.hubert) UNet_Conv=$($case.conv) UNet_Transpose=$($case.transpose)"
            $argsArray = @('infer-wav-full-benchmark',$Voice,$Hubert,$Rmvpe,$Index,$InputWav,$wav,
                $IndexRate.ToString($culture),'cuda',[string]$WarmRuns,[string]$CoreFrames,[string]$TensorBatchSize)
            $psi = [System.Diagnostics.ProcessStartInfo]::new()
            $psi.FileName = $exe
            $psi.WorkingDirectory = $PSScriptRoot
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $psi.RedirectStandardOutput = $true
            $psi.RedirectStandardError = $true
            $psi.Arguments = (@($argsArray | ForEach-Object { '"' + $_.Replace('"','\"') + '"' }) -join ' ')
            $proc = [System.Diagnostics.Process]::new()
            $proc.StartInfo = $psi
            try {
                if (-not $proc.Start()) { throw "Cannot launch $name" }
                $outTask = $proc.StandardOutput.ReadToEndAsync()
                $errTask = $proc.StandardError.ReadToEndAsync()
                $proc.WaitForExit()
                [IO.File]::WriteAllText($stdout,$outTask.GetAwaiter().GetResult(),[Text.Encoding]::UTF8)
                [IO.File]::WriteAllText($stderr,$errTask.GetAwaiter().GetResult(),[Text.Encoding]::UTF8)
                if ($proc.ExitCode -ne 0) { throw "$name failed with exit code $($proc.ExitCode)" }
            } finally { $proc.Dispose() }
            $report = Get-Content -LiteralPath "$wav.benchmark.json" -Raw | ConvertFrom-Json
            $expectedConv = if ($case.conv -and $case.transpose) {'implicit_gemm'} elseif ($case.conv) {'conv2d_implicit_gemm'} elseif ($case.transpose) {'transpose2d_gemm'} else {'direct'}
            $expectedNsf = if ($case.nsf -eq 'gemm') {'implicit_gemm'} else {'direct'}
            $expectedTemporal = if ($case.nsf -in @('quad','quad_interior')) {$case.nsf} else {''}
            $expectedHubert = if ($case.hubert) {'implicit_gemm'} else {'direct'}
            if ($report.stage -ne 63 -or $report.backend -ne 'cuda' -or $report.graph_replay -ne $false -or
                $report.inference_ms.Count -ne $WarmRuns -or $report.model_input_seconds -lt 10 -or
                $report.nsf_conv_execution -ne $expectedNsf -or $report.nsf_temporal_conv_kernel -ne $expectedTemporal -or
                $report.rmvpe_gru_execution -ne 'persistent' -or $report.rmvpe_unet_conv_execution -ne $expectedConv -or
                $report.hubert_frontend_conv_execution -ne $expectedHubert -or $report.iterations_exact_f32 -ne $true) {
                throw "${name}: invalid execution metadata or non-deterministic output"
            }
            $rows += [pscustomobject]@{
                mode=$name; group=$case.group; nsf=$case.nsf;
                unet_conv2d_gemm=$case.conv; unet_transpose2d_gemm=$case.transpose;
                hubert_gemm=$case.hubert; first_ms=$report.first_inference_ms;
                warm_ms=$report.warm_median_ms; warm_rtf=$report.warm_rtf;
                output_samples=$report.output_samples; chunks=$report.chunk_counts[0];
                sha256=(Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash
            }
            Write-Host ("  warm={0:N2}ms RTF={1:N4}" -f $report.warm_median_ms,$report.warm_rtf)
        } catch {
            $errors += [pscustomobject]@{mode=$name;error=$_.Exception.Message}
            Write-Warning "Stage73 $name failed: $($_.Exception.Message)"
            $_.Exception.ToString() | Set-Content -LiteralPath (Join-Path $root "$name.error.txt")
        }
    }
    $parity = @()
    if (@($rows | Where-Object mode -eq 'baseline').Count -eq 1) {
        $reference = [IO.File]::ReadAllBytes((Join-Path $root 'baseline.wav'))
        foreach ($row in @($rows | Where-Object mode -ne 'baseline')) {
            $candidate = [IO.File]::ReadAllBytes((Join-Path $root "$($row.mode).wav"))
            if ($reference.Length -ne $candidate.Length -or $reference.Length -ne 44 + $row.output_samples*2) {
                throw "PCM16 output layout mismatch: $($row.mode)"
            }
            $sampleCount = [int]$row.output_samples
            $a = [int16[]]::new($sampleCount)
            $b = [int16[]]::new($sampleCount)
            [Buffer]::BlockCopy($reference,44,$a,0,$sampleCount*2)
            [Buffer]::BlockCopy($candidate,44,$b,0,$sampleCount*2)
            $max = 0; $sum = 0.0; $different = 0
            for ($i=0;$i -lt $sampleCount;$i++) {
                $delta = [int]$a[$i]-[int]$b[$i]
                if ($delta -ne 0) { $different++ }
                $max = [Math]::Max($max,[Math]::Abs($delta))
                $sum += [double]$delta*$delta
            }
            $rmse = [Math]::Sqrt($sum/$sampleCount)
            $parity += [pscustomobject]@{
                mode=$row.mode; samples=$sampleCount; unequal=$different;
                rmse_pcm16=$rmse; max_abs_pcm16=$max;
                provisional_quality_gate=($rmse -le 5 -and $max -le 128)
            }
        }
    }
    [pscustomobject]@{
        stage=73; suite=$Suite; backend='cuda'; graph_replay=$false;
        source='Full real WAV, 32 chunks, no CUDA Graph, persistent RMVPE GRU';
        cases=$rows; parity=$parity; failures=$errors
    } | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Stage73 full WAV ZIP: $zip"
}
