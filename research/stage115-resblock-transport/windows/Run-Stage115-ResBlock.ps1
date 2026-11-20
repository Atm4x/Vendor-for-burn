param(
    [ValidateSet('measure','ncu','nsys','all')][string]$Profiler = 'measure',
    [string]$Modes = 'baseline,gemm,transport',
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
    [string]$NcuExe = 'ncu',
    [string]$NsysExe = 'nsys',
    [string]$OutDir = '.\stage115_results'
)
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$reports = Join-Path $OutDir $runId
New-Item -ItemType Directory -Force -Path $reports | Out-Null
$exe = (Resolve-Path -LiteralPath $RustExe).Path
$voicePath = (Resolve-Path -LiteralPath $Voice).Path
$hubertPath = (Resolve-Path -LiteralPath $Hubert).Path
$rmvpePath = (Resolve-Path -LiteralPath $Rmvpe).Path
$wavPath = (Resolve-Path -LiteralPath $InputWav).Path
$indexPath = if ($Index -eq '-') { '-' } else { (Resolve-Path -LiteralPath $Index).Path }
$old = @{}
foreach ($name in @('RVC_NSF_RESBLOCK_TRANSPORT','RVC_NVTX','RVC_NSF_RESBLOCK_FP16',
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
try {
    foreach ($name in @('RVC_NSF_RESBLOCK_FP16','RVC_NSF_PAIR_SHARED_TILE',
        'RVC_NSF_FUSED_PAIR_MAX_CHANNELS','RVC_NSF_CONV_KERNEL',
        'RVC_NSF_IMPLICIT_GEMM','RVC_NSF_CONV1D_GEMM')) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    $env:RVC_NVTX = '1'
    $nvtxStatus = & $exe nvtx-status
    $nvtxAvailable = "$nvtxStatus" -match 'nvtx_available=true'
    $selected = @($Modes.Split(',') | ForEach-Object { $_.Trim().ToLowerInvariant() })
    foreach ($mode in $selected) {
        if ($mode -notin @('baseline','gemm','transport')) { throw "Unsupported mode: $mode" }
        Remove-Item Env:\RVC_NSF_RESBLOCK_TRANSPORT -ErrorAction SilentlyContinue
        Remove-Item Env:\RVC_NSF_IMPLICIT_GEMM -ErrorAction SilentlyContinue
        if ($mode -eq 'transport') { $env:RVC_NSF_RESBLOCK_TRANSPORT = '1' }
        if ($mode -eq 'gemm') { $env:RVC_NSF_IMPLICIT_GEMM = '1' }
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
            foreach ($range in @('stage115_resblock_first','stage115_resblock_second')) {
                $targetPrefix = Join-Path $reports "$mode.$range"
                $ncuArgs = @('--target-processes','all','--nvtx','--nvtx-include',$range,
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
    $summary = [pscustomobject]@{
        stage=115; device='local Windows CUDA'; profiler=$Profiler; modes=$selected; rust_exe=$exe
        input_wav=$wavPath; duration_seconds=$DurationSeconds; runs=$Runs
        ncu_launch_skip=$NcuLaunchSkip; ncu_launch_count=$NcuLaunchCount
        nvtx_available=$nvtxAvailable; results=$results
        important='Profiler replay times are NOT comparable to uninstrumented warm medians.'
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reports 'manifest.json') -Encoding UTF8
    Write-Host "Stage115 results: $reports"
    Write-Host ($summary | ConvertTo-Json -Depth 6)
} finally {
    foreach ($name in $old.Keys) { [Environment]::SetEnvironmentVariable($name, $old[$name], 'Process') }
}
