param(
    [ValidateRange(1,6)][int]$Cycles = 2,
    [ValidateRange(2,30)][int]$Runs = 5,
    [ValidateSet('measure','ncu','nsys','all')][string]$Profiler = 'all',
    [ValidateRange(0,10000)][double]$StartSeconds = 0,
    [ValidateRange(0.2,9)][double]$DurationSeconds = 3,
    [string]$RustExe = '.\tts-with-rvc-rs.exe',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [string]$OutDir = '.\stage120_results',
    [string]$NcuExe = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe',
    [string]$NsysExe = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe',
    [string]$NvtxDllDir = 'C:\Users\mrtak\miniconda3\envs\torch2.6.0\Lib\site-packages\torch\lib',
    [ValidateRange(0,100000)][int]$NcuLaunchSkip = 34,
    [ValidateRange(1,10)][int]$NcuLaunchCount = 2,
    [switch]$NoPersistentGru,
    [switch]$KeepWav,
    [switch]$CheckOnly
)
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture
$root = $PSScriptRoot
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
function Resolve-Input([string]$Path) {
    $candidate = if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $root $Path }
    return (Resolve-Path -LiteralPath $candidate -ErrorAction Stop).Path
}
$exe = Resolve-Input $RustExe
$voiceFile = Resolve-Input $Voice
$hubertFile = Resolve-Input $Hubert
$rmvpeFile = Resolve-Input $Rmvpe
$inputFile = Resolve-Input $InputWav
$indexFile = if ($Index -eq '-') { '-' } else { Resolve-Input $Index }
$outputRoot = if ([IO.Path]::IsPathRooted($OutDir)) { [IO.Path]::GetFullPath($OutDir) }
    else { [IO.Path]::GetFullPath((Join-Path $root $OutDir)) }
$reports = Join-Path $outputRoot $stamp
if ($Profiler -in @('ncu','all')) { $NcuExe = Resolve-Input $NcuExe }
if ($Profiler -in @('nsys','all')) { $NsysExe = Resolve-Input $NsysExe }
if ($Profiler -in @('nsys','all')) {
    if (-not (Test-Path (Join-Path $NvtxDllDir 'nvToolsExt64_1.dll') -PathType Leaf)) {
        throw "Required installed NVTX DLL not found: $NvtxDllDir"
    }
}
Write-Host "Stage120: prefetch vs original temporal transport, Rust=$exe"
Write-Host "Output: $reports"
Write-Host "Persistent GRU enabled: $(-not $NoPersistentGru)"
if ($CheckOnly) { Write-Host 'Stage120 preflight PASS; no inference executed.'; return }
$managedEnv = @(
    'RVC_NSF_RESBLOCK_TRANSPORT','RVC_NSF_RESBLOCK_SHARED_NTC',
    'RVC_NSF_RESBLOCK_TEMPORAL_TILE','RVC_NSF_RESBLOCK_WEIGHT_PREFETCH',
    'RVC_RMVPE_GRU_PERSISTENT','RVC_NSF_IMPLICIT_GEMM',
    'RVC_NSF_RESBLOCK_FP16','RVC_NSF_PAIR_SHARED_TILE',
    'RVC_NSF_FUSED_PAIR_MAX_CHANNELS','RVC_NSF_CONV_KERNEL',
    'RVC_NSF_CONV1D_GEMM','RVC_NSF_TILE_CORE_FRAMES',
    'RVC_GPU_F0','RVC_GPU_FAISS','RVC_NVTX'
)
$before = @{}
foreach ($key in $managedEnv) { $before[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$oldPath = $env:PATH
$results = [System.Collections.Generic.List[object]]::new()
New-Item -ItemType Directory -Force -Path $reports | Out-Null
function Configure-Mode([string]$Mode,[bool]$UseNvtx) {
    foreach ($key in $managedEnv) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    if (-not $NoPersistentGru) { $env:RVC_RMVPE_GRU_PERSISTENT = '1' }
    if ($Mode -ne 'baseline') {
        $env:RVC_NSF_RESBLOCK_TRANSPORT = '1'
        $env:RVC_NSF_RESBLOCK_SHARED_NTC = '1'
        $env:RVC_NSF_RESBLOCK_TEMPORAL_TILE = '4'
        if ($Mode -eq 'prefetch') { $env:RVC_NSF_RESBLOCK_WEIGHT_PREFETCH = '1' }
    }
    $env:RVC_NVTX = if ($UseNvtx) { '1' } else { '0' }
}
function Run-Logged([string]$Program,[string[]]$Arguments,[string]$Prefix) {
    $quoted = ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + $_.Replace('"','\"') + '"' } else { $_ }
    }) -join ' '
    $stdout = "$Prefix.stdout.log"
    $stderr = "$Prefix.stderr.log"
    $proc = Start-Process -FilePath $Program -ArgumentList $quoted -WorkingDirectory $root -Wait -NoNewWindow -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($proc.ExitCode -ne 0) {
        Get-Content -LiteralPath $stderr -Tail 35 -ErrorAction SilentlyContinue | Write-Host
        throw "Stage120 process exited $($proc.ExitCode): $Program; logs: $stdout / $stderr"
    }
}
function Get-PcmDelta([string]$Reference,[string]$Candidate) {
    $a = [IO.File]::ReadAllBytes([IO.Path]::GetFullPath($Reference))
    $b = [IO.File]::ReadAllBytes([IO.Path]::GetFullPath($Candidate))
    if ($a.Length -ne $b.Length -or $a.Length -lt 44 -or
        [Text.Encoding]::ASCII.GetString($a,0,4) -ne 'RIFF' -or
        [Text.Encoding]::ASCII.GetString($b,0,4) -ne 'RIFF' -or
        [Text.Encoding]::ASCII.GetString($a,8,4) -ne 'WAVE' -or
        [Text.Encoding]::ASCII.GetString($b,8,4) -ne 'WAVE' -or
        [Text.Encoding]::ASCII.GetString($a,36,4) -ne 'data' -or
        [Text.Encoding]::ASCII.GetString($b,36,4) -ne 'data' -or
        [BitConverter]::ToInt16($a,20) -ne 1 -or [BitConverter]::ToInt16($b,20) -ne 1 -or
        [BitConverter]::ToInt16($a,22) -ne 1 -or [BitConverter]::ToInt16($b,22) -ne 1 -or
        [BitConverter]::ToInt16($a,34) -ne 16 -or [BitConverter]::ToInt16($b,34) -ne 16 -or
        [BitConverter]::ToInt32($a,24) -ne [BitConverter]::ToInt32($b,24) -or
        [BitConverter]::ToInt32($a,40) -ne ($a.Length - 44) -or
        [BitConverter]::ToInt32($b,40) -ne ($b.Length - 44)) {
        throw 'Stage120 parity requires matched 44-byte PCM16 mono WAVs.'
    }
    $n = [int](($a.Length - 44) / 2)
    [double]$sumD = 0; [double]$sumS = 0; [int]$changed = 0; [int]$max = 0
    for ($i = 0; $i -lt $n; $i++) {
        $x = [int][BitConverter]::ToInt16($a,44 + 2*$i)
        $y = [int][BitConverter]::ToInt16($b,44 + 2*$i)
        $d = $x - $y
        if ($d -ne 0) { $changed++ }
        if ([Math]::Abs($d) -gt $max) { $max = [Math]::Abs($d) }
        $sumD += [double]$d*$d; $sumS += [double]$x*$x
    }
    [pscustomobject]@{
        samples=$n; different_samples=$changed; max_abs_pcm_lsb=$max
        normalized_rmse=[Math]::Sqrt($sumD/$n)/32767.0
        snr_db=$(if ($sumD -eq 0) { 'infinite' } elseif ($sumS -eq 0) { '-infinite' }
            else { 10.0*[Math]::Log10($sumS/$sumD) })
    }
}
function Pack-Results([string]$Dir) {
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archivePath = "$Dir.zip"
    if (Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath -Force }
    $archive = [IO.Compression.ZipFile]::Open($archivePath,[IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($f in (Get-ChildItem -LiteralPath $Dir -File -Recurse)) {
            if ($f.Extension.ToLowerInvariant() -in @('.wav','.pth','.pt','.index','.onnx','.flac','.mp3','.safetensors','.dll','.exe')) { continue }
            $rel = $f.FullName.Substring($Dir.TrimEnd('\','/').Length).TrimStart('\','/').Replace('\','/')
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$f.FullName,$rel,[IO.Compression.CompressionLevel]::Optimal)
        }
    } finally { $archive.Dispose() }
    return $archivePath
}
function Get-BenchmarkArguments([string]$OutputWav,[int]$Repeat) {
    return [string[]]@('infer-wav-segment-benchmark',$voiceFile,$hubertFile,$rmvpeFile,
        $indexFile,$inputFile,$OutputWav,$StartSeconds.ToString($inv),
        $DurationSeconds.ToString($inv),$(if ($indexFile -eq '-') { '0' } else { '0.75' }),
        '0','cuda',$Repeat.ToString($inv))
}
try {
    $forward = @('baseline','control','prefetch')
    for ($cycle = 1; $cycle -le $Cycles; $cycle++) {
        $order = if (($cycle % 2) -eq 1) { $forward } else { @($forward[2],$forward[1],$forward[0]) }
        foreach ($mode in $order) {
            Configure-Mode $mode $false
            $prefix = Join-Path $reports ("cycle{0}_{1}" -f $cycle,$mode)
            $wav = "$prefix.wav"
            Write-Host "Stage120 measure: $cycle/$Cycles $mode"
            Run-Logged $exe (Get-BenchmarkArguments $wav $Runs) $prefix
            $benchPath = "$wav.benchmark.json"
            if (-not (Test-Path -LiteralPath $benchPath -PathType Leaf)) { throw "Benchmark JSON missing: $benchPath" }
            $bench = Get-Content -LiteralPath $benchPath -Raw | ConvertFrom-Json
            if ($bench.inference_ms.Count -ne $Runs -or $bench.warm_median_ms -le 0) {
                throw "Stage120 invalid benchmark: $benchPath"
            }
            $results.Add([pscustomobject]@{
                cycle=$cycle; mode=$mode; warm_median_ms=[double]$bench.warm_median_ms
                cold_ms=[double]$bench.inference_ms[0]; samples=[int]$bench.output_samples
                pcm_sha256=(Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash
                pcm_to_baseline=$null; pcm_to_control=$null; wav_path=$wav
            })
        }
    }
    for ($cycle = 1; $cycle -le $Cycles; $cycle++) {
        $base = @($results | Where-Object { $_.cycle -eq $cycle -and $_.mode -eq 'baseline' })[0]
        $control = @($results | Where-Object { $_.cycle -eq $cycle -and $_.mode -eq 'control' })[0]
        foreach ($result in @($results | Where-Object { $_.cycle -eq $cycle })) {
            $result.pcm_to_baseline = Get-PcmDelta $base.wav_path $result.wav_path
            $result.pcm_to_control = Get-PcmDelta $control.wav_path $result.wav_path
        }
    }
    if ($Profiler -in @('ncu','nsys','all')) {
        foreach ($mode in @('control','prefetch')) {
            Configure-Mode $mode $false
            if ($Profiler -in @('ncu','all')) {
                $profileWav = Join-Path $reports "$mode.ncu-temp.wav"
                $prefix = Join-Path $reports "$mode.ncu"
                $ncuArgs = [string[]]@('--target-processes','all',
                    '--kernel-name','regex:preact_residual_temporal_kernel',
                    '--launch-skip',$NcuLaunchSkip.ToString($inv),
                    '--launch-count',$NcuLaunchCount.ToString($inv),
                    '--set','full','--force-overwrite','--export',"$prefix.ncu-rep",$exe)
                $ncuArgs += Get-BenchmarkArguments $profileWav 2
                Write-Host "Stage120 NCU: $mode"
                Run-Logged $NcuExe $ncuArgs $prefix
            }
            if ($Profiler -in @('nsys','all')) {
                $env:PATH = "$NvtxDllDir;$oldPath"
                $env:RVC_NVTX = '1'
                $profileWav = Join-Path $reports "$mode.nsys-temp.wav"
                $prefix = Join-Path $reports "$mode.nsys"
                $nsysArgs = [string[]]@('profile','--trace=cuda,nvtx','--sample=none',
                    '--force-overwrite=true','--output',$prefix,$exe)
                $nsysArgs += Get-BenchmarkArguments $profileWav 2
                Write-Host "Stage120 NSYS: $mode"
                Run-Logged $NsysExe $nsysArgs $prefix
                $rep = "$prefix.nsys-rep"
                if (-not (Test-Path -LiteralPath $rep -PathType Leaf)) { throw "NSYS report missing: $rep" }
                $sqlite = "$prefix.sqlite"
                Write-Host "Stage120 NSYS full SQLite timeline: $mode"
                Run-Logged $NsysExe ([string[]]@('export','--type','sqlite','--force-overwrite=true','--output',$sqlite,$rep)) "$prefix.export"
                $env:PATH = $oldPath
            }
        }
    }
    $public = @($results | ForEach-Object {
        [pscustomobject]@{
            cycle=$_.cycle; mode=$_.mode; warm_median_ms=$_.warm_median_ms; cold_ms=$_.cold_ms
            samples=$_.samples; pcm_sha256=$_.pcm_sha256
            pcm_to_baseline=$_.pcm_to_baseline; pcm_to_control=$_.pcm_to_control
        }
    })
    $manifest = [pscustomobject]@{
        stage=120; rust_feature='RVC_NSF_RESBLOCK_WEIGHT_PREFETCH=1'
        design='alternating independent full-RVC processes; baseline/control/prefetch'
        geometry='C128 K11, temporal tile 4, prefetch next input-channel weight'
        persistent_gru=(-not $NoPersistentGru)
        duration_seconds=$DurationSeconds; cycles=$Cycles; runs=$Runs
        profiler=$Profiler; ncu_skip=$NcuLaunchSkip; ncu_count=$NcuLaunchCount
        nvtx_capture=$($Profiler -in @('nsys','all'))
        benchmark_results=$public
        note='NCU replay wall time must not be compared with uninstrumented warm_median_ms. Validate launch geometry of captured kernels.'
    }
    $manifest | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath (Join-Path $reports 'manifest.json') -Encoding UTF8
    $public | Select-Object cycle,mode,warm_median_ms,cold_ms,samples,pcm_sha256 |
        Export-Csv -LiteralPath (Join-Path $reports 'summary.csv') -NoTypeInformation -Encoding UTF8
    $public | Format-Table cycle,mode,warm_median_ms,cold_ms -AutoSize
    Write-Host 'Stage120 benchmark, PCM and profiling completed.'
} finally {
    if (Test-Path -LiteralPath $reports) {
        try { Write-Host "Stage120 results ZIP (no WAV/models): $(Pack-Results $reports)" }
        catch { Write-Warning "Stage120 archiving failed: $_" }
        if (-not $KeepWav) {
            Get-ChildItem -LiteralPath $reports -File -Filter '*.wav' |
                Remove-Item -Force -ErrorAction SilentlyContinue
        }
    }
    foreach ($key in $managedEnv) { [Environment]::SetEnvironmentVariable($key,$before[$key],'Process') }
    $env:PATH = $oldPath
}
