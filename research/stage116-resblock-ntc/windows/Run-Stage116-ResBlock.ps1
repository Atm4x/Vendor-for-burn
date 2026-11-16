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
    [switch]$KeepWav,
    [switch]$NoArchive,
    [string]$OutDir = '.\stage116_results'
)
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$oldPath = $env:PATH
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
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
function Find-Executable([string]$Name) {
    if (Test-Path -LiteralPath $Name -PathType Leaf) { return (Resolve-Path -LiteralPath $Name).Path }
    $command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $file = [IO.Path]::GetFileName($Name)
    if (-not $file.EndsWith('.exe', [StringComparison]::OrdinalIgnoreCase)) { $file += '.exe' }
    $roots = [System.Collections.Generic.List[string]]::new()
    foreach ($envName in @('CUDA_PATH', 'NVTOOLSEXT_PATH')) {
        $value = [Environment]::GetEnvironmentVariable($envName)
        if ($value) { $roots.Add($value) }
    }
    foreach ($var in (Get-ChildItem Env:CUDA_PATH* -ErrorAction SilentlyContinue)) {
        if ($var.Value) { $roots.Add($var.Value) }
    }
    foreach ($base in @($env:ProgramFiles, [Environment]::GetEnvironmentVariable('ProgramFiles(x86)'))) {
        if ($base) {
            $roots.Add((Join-Path $base 'NVIDIA Corporation'))
            $roots.Add((Join-Path $base 'NVIDIA GPU Computing Toolkit\CUDA'))
        }
    }
    foreach ($root in ($roots | Select-Object -Unique)) {
        if (-not (Test-Path -LiteralPath $root -PathType Container)) { continue }
        $hit = Get-ChildItem -LiteralPath $root -Filter $file -File -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    throw "$file not found in PATH or installed NVIDIA/CUDA/Nsight directories. Specify its full path with -NcuExe or -NsysExe."
}
function Find-NvtxDll {
    $file = 'nvToolsExt64_1.dll'
    $direct = [System.Collections.Generic.List[string]]::new()
    $roots = [System.Collections.Generic.List[string]]::new()
    if ($NvtxDllDir) {
        if (-not (Test-Path -LiteralPath $NvtxDllDir -PathType Container)) {
            throw "NvtxDllDir does not exist: $NvtxDllDir"
        }
        $direct.Add((Resolve-Path -LiteralPath $NvtxDllDir).Path)
    }
    $direct.Add($scriptRoot)
    $direct.Add((Split-Path -Parent $exe))
    foreach ($var in (Get-ChildItem Env:CUDA_PATH* -ErrorAction SilentlyContinue)) {
        if ($var.Value) {
            $roots.Add($var.Value)
            $direct.Add((Join-Path $var.Value 'bin'))
            $direct.Add((Join-Path $var.Value 'extras\NVTX\bin'))
            $direct.Add((Join-Path $var.Value 'extras\NVTX\lib\x64'))
        }
    }
    if ($env:NVTOOLSEXT_PATH) {
        $roots.Add($env:NVTOOLSEXT_PATH)
        $direct.Add($env:NVTOOLSEXT_PATH)
        $direct.Add((Join-Path $env:NVTOOLSEXT_PATH 'bin\x64'))
    }
    foreach ($base in @($env:ProgramFiles, [Environment]::GetEnvironmentVariable('ProgramFiles(x86)'))) {
        if (-not $base) { continue }
        $nvidia = Join-Path $base 'NVIDIA Corporation'
        $cuda = Join-Path $base 'NVIDIA GPU Computing Toolkit\CUDA'
        $roots.Add($nvidia)
        $roots.Add($cuda)
        $direct.Add((Join-Path $nvidia 'NvToolsExt\bin\x64'))
    }
    foreach ($dir in ($env:PATH -split ';')) {
        if ($dir) { $direct.Add($dir.Trim('"')) }
    }
    foreach ($dir in ($direct | Select-Object -Unique)) {
        if (-not $dir) { continue }
        $dll = Join-Path $dir $file
        if (Test-Path -LiteralPath $dll -PathType Leaf) { return (Resolve-Path -LiteralPath $dll).Path }
    }
    foreach ($root in ($roots | Select-Object -Unique)) {
        if (-not $root -or -not (Test-Path -LiteralPath $root -PathType Container)) { continue }
        $dll = Get-ChildItem -LiteralPath $root -File -Filter $file -Recurse -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($dll) { return $dll.FullName }
    }
    return $null
}
function Ensure-Nvtx {
    $status = & $exe nvtx-status
    if ($LASTEXITCODE -eq 0 -and "$status" -match 'nvtx_available=true') {
        Write-Host 'NVTX already available to Rust executable.'
        return $true
    }
    $dll = Find-NvtxDll
    if ($dll) {
        $dir = Split-Path -Parent $dll
        $env:PATH = "$dir;$env:PATH"
        Write-Host "Using installed NVTX DLL: $dll"
        $status = & $exe nvtx-status
        if ($LASTEXITCODE -eq 0 -and "$status" -match 'nvtx_available=true') { return $true }
        Write-Warning "NVTX DLL was found but Rust reported unavailable: $status"
    } else {
        Write-Warning 'nvToolsExt64_1.dll not found under installed NVIDIA/CUDA/Nsight locations or PATH.'
    }
    Write-Warning 'NCU will run without NVTX filtering; NSYS will capture CUDA events. This is not a ResBlock-specific NCU profile.'
    return $false
}
function Save-ProfileArchive([string]$ReportDir) {
    $zipPath = "$ReportDir.zip"
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    $zip = [IO.Compression.ZipFile]::Open($zipPath, [IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($file in (Get-ChildItem -LiteralPath $ReportDir -Recurse -File)) {
            if ($file.Extension -in @('.wav','.pth','.pt','.index','.flac','.mp3','.onnx')) { continue }
            $relative = $file.FullName.Substring($ReportDir.TrimEnd('\','/').Length).TrimStart('\','/').Replace('\','/')
            [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $file.FullName, $relative, [IO.Compression.CompressionLevel]::Optimal)
        }
    } finally { $zip.Dispose() }
    return $zipPath
}
try {
    foreach ($name in @('RVC_NSF_RESBLOCK_FP16','RVC_NSF_PAIR_SHARED_TILE',
        'RVC_NSF_FUSED_PAIR_MAX_CHANNELS','RVC_NSF_CONV_KERNEL',
        'RVC_NSF_IMPLICIT_GEMM','RVC_NSF_CONV1D_GEMM')) {
        [Environment]::SetEnvironmentVariable($name, $null, 'Process')
    }
    $env:RVC_NVTX = '1'
    Remove-Item Env:\RVC_NSF_RESBLOCK_SHARED_NTC -ErrorAction SilentlyContinue
    if ($Profiler -in @('ncu','all')) { $NcuExe = Find-Executable $NcuExe }
    if ($Profiler -in @('nsys','all')) { $NsysExe = Find-Executable $NsysExe }
    $nvtxAvailable = if ($Profiler -eq 'measure') { $false } else { Ensure-Nvtx }
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
            $ranges = if ($nvtxAvailable) { @(
                ('stage116_resblock_c{0}_k{1}_d{2}_first/' -f $ProfileChannels,$ProfileKernel,$ProfileDilation),
                ('stage116_resblock_c{0}_k{1}_d1_second/' -f $ProfileChannels,$ProfileKernel)
            ) } else { @('unfiltered') }
            foreach ($range in $ranges) {
                $rangeName = $range.TrimEnd('/')
                $targetPrefix = Join-Path $reports "$mode.$rangeName"
                $ncuArgs = @('--target-processes','all',
                    '--set','full','--launch-skip',$NcuLaunchSkip.ToString($inv),
                    '--launch-count',$NcuLaunchCount.ToString($inv),
                    '--force-overwrite','--export',"$targetPrefix.ncu-rep")
                if ($nvtxAvailable) { $ncuArgs += @('--nvtx','--nvtx-push-pop-scope','process','--nvtx-include',$range) }
                $ncuArgs += @($exe) + $args
                Invoke-Captured $NcuExe $ncuArgs "$targetPrefix.ncu"
            }
        }
        if ($Profiler -eq 'nsys' -or $Profiler -eq 'all') {
            $nsysPrefix = Join-Path $reports "$mode.nsys"
            $nsysTrace = if ($nvtxAvailable) { 'cuda,nvtx,osrt' } else { 'cuda,osrt' }
            $nsysArgs = @('profile',"--trace=$nsysTrace",'--sample=none',
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
        nvtx_available=$nvtxAvailable; nvtx_scoped_ncu=$nvtxAvailable; results=$results
        important='Profiler replay times are NOT comparable to uninstrumented warm medians.'
    }
    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reports 'manifest.json') -Encoding UTF8
    if (-not $NoArchive) {
        $archive = Save-ProfileArchive $reports
        Write-Host "Stage116 profile archive (no WAV/model assets): $archive"
    }
    Write-Host "Stage116 results: $reports"
    Write-Host ($summary | ConvertTo-Json -Depth 6)
} finally {
    if (-not $KeepWav -and (Test-Path -LiteralPath $reports)) {
        Get-ChildItem -LiteralPath $reports -Filter '*.wav' -File | Remove-Item -Force -ErrorAction SilentlyContinue
    }
    foreach ($name in $old.Keys) { [Environment]::SetEnvironmentVariable($name, $old[$name], 'Process') }
    $env:PATH = $oldPath
}
