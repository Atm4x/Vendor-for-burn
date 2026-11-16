param(
    [ValidateSet('measure','ncu','nsys','all')][string]$Profiler = 'measure',
    [string]$Modes = 'baseline,baseline_ntc,gemm,gemm_ntc,transport,transport_ntc',
    [string]$RustExe = '.\tts-with-rvc-rs.exe',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(0,10000)][double]$StartSeconds = 0,
    [ValidateRange(0.2,9.0)][double]$DurationSeconds = 3,
    [ValidateRange(2,32)][int]$Runs = 5,
    [ValidateRange(1,32)][int]$NcuLaunchCount = 6,
    [ValidateRange(0,10000)][int]$NcuLaunchSkip = 0,
    [ValidateRange(1,4096)][int]$ProfileChannels = 128,
    [ValidateSet(3,7,11)][int]$ProfileKernel = 11,
    [ValidateRange(1,32)][int]$ProfileDilation = 5,
    [string]$ReferenceWav = '',
    [string]$NcuExe = 'ncu',
    [string]$NsysExe = 'nsys',
    [string]$NvtxDllDir = '',
    [string]$OutDir = '.\stage116_results'
)
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$oldPath = $env:PATH
if ($NvtxDllDir) {
    $dllDir = (Resolve-Path -LiteralPath $NvtxDllDir).Path
    if (-not (Test-Path (Join-Path $dllDir 'nvToolsExt64_1.dll'))) {
        throw 'nvToolsExt64_1.dll not found in NvtxDllDir'
    }
    $env:PATH = "$dllDir;$env:PATH"
}
$reports = Join-Path $OutDir $runId
New-Item -ItemType Directory -Force -Path $reports | Out-Null
$exe = (Resolve-Path -LiteralPath $RustExe).Path
$voicePath = (Resolve-Path -LiteralPath $Voice).Path
$hubertPath = (Resolve-Path -LiteralPath $Hubert).Path
$rmvpePath = (Resolve-Path -LiteralPath $Rmvpe).Path
$wavPath = (Resolve-Path -LiteralPath $InputWav).Path
$indexPath = if ($Index -eq '-') { '-' } else { (Resolve-Path -LiteralPath $Index).Path }
$old = @{}
foreach ($name in @('RVC_NSF_RESBLOCK_TRANSPORT','RVC_NSF_RESBLOCK_SHARED_NTC','RVC_NVTX','RVC_NSF_RESBLOCK_FP16',
    'RVC_NSF_PAIR_SHARED_TILE','RVC_NSF_FUSED_PAIR_MAX_CHANNELS',
    'RVC_NSF_CONV_KERNEL','RVC_NSF_IMPLICIT_GEMM','RVC_NSF_CONV1D_GEMM')) {
    $old[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$results = New-Object System.Collections.Generic.List[object]
function Invoke-Captured([string]$File, [string[]]$Arguments, [string]$Prefix) {
    $stdout = "$Prefix.stdout.log"
    $stderr = "$Prefix.stderr.log"
    $quoted = ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"', '\"') + '"' } else { $_ }
    }) -join ' '
    $process = Start-Process -FilePath $File -ArgumentList $quoted -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($process.ExitCode -ne 0) {
        Get-Content -LiteralPath $stderr -Tail 40 | ForEach-Object { Write-Host $_ }
        throw "Exit $($process.ExitCode): $File; logs: $stdout, $stderr"
    }
}
function Compare-Pcm16([string]$Source, [string]$Candidate) {
    $a = [IO.File]::ReadAllBytes($Source)
    $b = [IO.File]::ReadAllBytes($Candidate)
    foreach ($bytes in @($a,$b)) {
        if ($bytes.Length -lt 44 -or
            [Text.Encoding]::ASCII.GetString($bytes, 0, 4) -ne 'RIFF' -or
            [Text.Encoding]::ASCII.GetString($bytes, 8, 4) -ne 'WAVE' -or
            [BitConverter]::ToInt16($bytes, 20) -ne 1 -or
            [BitConverter]::ToInt16($bytes, 22) -ne 1 -or
            [BitConverter]::ToInt16($bytes, 34) -ne 16 -or
            [Text.Encoding]::ASCII.GetString($bytes, 36, 4) -ne 'data') {
            throw "Expected a standard PCM16 mono RIFF WAV: $Source / $Candidate"
        }
    }
    $rateA = [BitConverter]::ToInt32($a, 24)
    $rateB = [BitConverter]::ToInt32($b, 24)
    $sizeA = [BitConverter]::ToInt32($a, 40)
    $sizeB = [BitConverter]::ToInt32($b, 40)
    if ($rateA -ne $rateB -or $sizeA -ne $sizeB -or
        $sizeA -ne ($a.Length - 44) -or $sizeB -ne ($b.Length - 44) -or
        $sizeA % 2 -ne 0) { throw 'PCM WAV sample rates, lengths, or payload sizes mismatch' }
    [double]$sumSq = 0
    [double]$signalSq = 0
    [int]$maxAbs = 0
    [int]$different = 0
    $count = $sizeA / 2
    for ($i = 0; $i -lt $count; $i++) {
        $x = [int][BitConverter]::ToInt16($a, 44 + 2 * $i)
        $y = [int][BitConverter]::ToInt16($b, 44 + 2 * $i)
        $error = [int]($x - $y)
        if ($error -ne 0) { $different++ }
        $abs = [Math]::Abs($error)
        if ($abs -gt $maxAbs) { $maxAbs = $abs }
        $sumSq += [double]$error * $error
        $signalSq += [double]$x * $x
    }
    $rmse = if ($count -gt 0) { [Math]::Sqrt($sumSq / $count) / 32767.0 } else { 0.0 }
    $snr = if ($sumSq -eq 0) { 'infinite' }
        elseif ($signalSq -eq 0) { '-infinite' }
        else { 10.0 * [Math]::Log10($signalSq / $sumSq) }
    return [pscustomobject]@{
        reference=$Source; samples=$count; rate=$rateA
        max_abs_pcm_lsb=$maxAbs; rmse_normalized=$rmse
        different_pcm_samples=$different; snr_db=$snr
    }
}
try {
    foreach ($name in @('RVC_NSF_RESBLOCK_FP16','RVC_NSF_PAIR_SHARED_TILE',
        'RVC_NSF_FUSED_PAIR_MAX_CHANNELS','RVC_NSF_CONV_KERNEL',
        'RVC_NSF_IMPLICIT_GEMM','RVC_NSF_CONV1D_GEMM')) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    $env:RVC_NVTX = '1'
    Remove-Item Env:\RVC_NSF_RESBLOCK_SHARED_NTC -ErrorAction SilentlyContinue
    $nvtxStatus = & $exe nvtx-status
    $nvtxAvailable = "$nvtxStatus" -match 'nvtx_available=true'
    $selected = @($Modes.Split(',') | ForEach-Object { $_.Trim().ToLowerInvariant() })
    foreach ($mode in $selected) {
        if ($mode -notin @('baseline','baseline_ntc','gemm','gemm_ntc','transport','transport_ntc')) {
            throw "Unsupported mode: $mode"
        }
        Remove-Item Env:\RVC_NSF_RESBLOCK_TRANSPORT -ErrorAction SilentlyContinue
        Remove-Item Env:\RVC_NSF_RESBLOCK_SHARED_NTC -ErrorAction SilentlyContinue
        Remove-Item Env:\RVC_NSF_IMPLICIT_GEMM -ErrorAction SilentlyContinue
        if ($mode.StartsWith('transport')) { $env:RVC_NSF_RESBLOCK_TRANSPORT = '1' }
        if ($mode.StartsWith('gemm')) { $env:RVC_NSF_IMPLICIT_GEMM = '1' }
        if ($mode.EndsWith('_ntc')) { $env:RVC_NSF_RESBLOCK_SHARED_NTC = '1' }
        $target = Join-Path $reports "$mode.wav"
        $args = @('infer-wav-segment-benchmark', $voicePath, $hubertPath, $rmvpePath,
            $indexPath, $wavPath, $target,
            $StartSeconds.ToString($inv), $DurationSeconds.ToString($inv),
            $(if ($indexPath -eq '-') { '0' } else { '0.75' }),'0','cuda', $Runs.ToString($inv))
        Invoke-Captured $exe $args (Join-Path $reports "$mode.measure")
        $hash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        $results.Add([pscustomobject]@{ mode=$mode; wav_sha256=$hash; bytes=(Get-Item $target).Length })
        if ($Profiler -eq 'ncu' -or $Profiler -eq 'all') {
            if (-not $nvtxAvailable) { throw 'NVTX DLL not available; install nvToolsExt64_1.dll in PATH' }
            $ranges = @(
                ('stage116_resblock_c{0}_k{1}_d{2}_first/' -f $ProfileChannels,$ProfileKernel,$ProfileDilation),
                ('stage116_resblock_c{0}_k{1}_d1_second/' -f $ProfileChannels,$ProfileKernel)
            )
            foreach ($range in $ranges) {
                $rangeName = $range.TrimEnd('/')
                $targetPrefix = Join-Path $reports "$mode.$rangeName"
                $ncuArgs = @('--target-processes','all','--nvtx','--nvtx-push-pop-scope','process',
                    '--nvtx-include',$range,
                    '--set','full','--launch-skip',$NcuLaunchSkip.ToString($inv),
                    '--launch-count',$NcuLaunchCount.ToString($inv),
                    '--force-overwrite','--export',"$targetPrefix.ncu-rep",$exe) + $args
                Invoke-Captured $NcuExe $ncuArgs "$targetPrefix.ncu"
            }
        }
        if ($Profiler -eq 'nsys' -or $Profiler -eq 'all') {
            if (-not $nvtxAvailable) { throw 'NVTX DLL not available; install nvToolsExt64_1.dll in PATH' }
            $nsysPrefix = Join-Path $reports "$mode.nsys"
            $nsysArgs = @('profile','--trace=cuda,nvtx,osrt','--sample=none',
                '--force-overwrite=true','--output', $nsysPrefix, $exe) + $args
            Invoke-Captured $NsysExe $nsysArgs (Join-Path $reports "$mode.nsys")
        }
    }
    $reference = if ($ReferenceWav) { (Resolve-Path -LiteralPath $ReferenceWav).Path }
        else { Join-Path $reports 'baseline.wav' }
    if (Test-Path -LiteralPath $reference) {
        foreach ($r in $results) {
            $wav = Join-Path $reports "$($r.mode).wav"
            $r | Add-Member -NotePropertyName parity -NotePropertyValue (Compare-Pcm16 $reference $wav)
        }
    }
    $summary = [pscustomobject]@{
        stage=116; device='local Windows CUDA'; profiler=$Profiler; modes=$selected; rust_exe=$exe
        input_wav=$wavPath; duration_seconds=$DurationSeconds; runs=$Runs
        ncu_launch_skip=$NcuLaunchSkip; ncu_launch_count=$NcuLaunchCount
        ncu_geometry=@{ channels=$ProfileChannels; kernel=$ProfileKernel; dilation=$ProfileDilation }
        nvtx_available=$nvtxAvailable; results=$results
        important='Profiler replay times are NOT comparable to uninstrumented warm medians.'
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reports 'manifest.json') -Encoding UTF8
    Write-Host "Stage116 results: $reports"
    Write-Host ($summary | ConvertTo-Json -Depth 6)
} finally {
    foreach ($name in $old.Keys) { [Environment]::SetEnvironmentVariable($name, $old[$name], 'Process') }
    $env:PATH = $oldPath
}
