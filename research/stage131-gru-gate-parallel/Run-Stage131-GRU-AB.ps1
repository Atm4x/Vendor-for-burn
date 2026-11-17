param(
    [ValidateRange(2,12)][int]$Runs = 3,
    [ValidateRange(1,4)][int]$Cycles = 2,
    [ValidateSet('measure','nsys','ncu','all')][string]$Profiler = 'all',
    [string]$RustExe = '.\tts-with-rvc-rs.exe',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [string]$OutDir = '.\stage131_gru_results',
    [string]$PythonExe = 'python',
    [ValidateRange(0,150)][double]$MinSnrDb = 70,
    [ValidateRange(25,900)][int]$CoreFrames = 300,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(0,1)][double]$IndexRate = 0.75,
    [string]$NsysExe = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe',
    [string]$NcuExe = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe',
    [string]$NvtxDllDir = 'C:\Users\mrtak\miniconda3\envs\torch2.6.0\Lib\site-packages\torch\lib',
    [switch]$CheckOnly,
    [switch]$KeepWav
)
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture
function FullPath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $Value))
}
$exe = FullPath $RustExe
$voicePath = FullPath $Voice
$hubertPath = FullPath $Hubert
$rmvpePath = FullPath $Rmvpe
$wavPath = FullPath $InputWav
$indexPath = if ($Index -eq '-') { '-' } else { FullPath $Index }
foreach ($path in @($exe,$voicePath,$hubertPath,$rmvpePath,$wavPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file $path" }
}
if ($IndexRate -gt 0 -and ($indexPath -eq '-' -or -not (Test-Path -LiteralPath $indexPath))) {
    throw "Missing FAISS index $indexPath"
}
$root = FullPath $OutDir
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ')
$work = Join-Path $root $stamp
$wavProbe = Join-Path $PSScriptRoot 'Validate-Stage131-WAV.py'
& $PythonExe $wavProbe $wavPath
if ($LASTEXITCODE -ne 0) { throw 'Stage131 requires a real 85-95 second WAV' }
if ($CheckOnly) { Write-Host "Stage131 preflight PASS; source=$wavPath executable=$exe"; return }
New-Item -ItemType Directory -Force -Path $work | Out-Null
$managed = @('RVC_NSF_IMPLICIT_GEMM','RVC_NSF_CONV1D_GEMM','RVC_NSF_CONV1D_GEMM_ASYNC',
    'RVC_NSF_TRANSPOSE_GEMM','RVC_NSF_RESBLOCK_TRANSPORT','RVC_NSF_RESBLOCK_SHARED_NTC',
    'RVC_NSF_RESBLOCK_FP16','RVC_NSF_PAIR_SHARED_TILE','RVC_NSF_FUSED_PAIR_MAX_CHANNELS',
    'RVC_RMVPE_GRU_PERSISTENT','RVC_RMVPE_GRU_WEIGHT_COALESCED','RVC_RMVPE_GRU_COLUMN_SPLIT','RVC_RMVPE_GRU_GATE_PARALLEL','RVC_RMVPE_UNET_GEMM',
    'RVC_RMVPE_UNET_COMPENSATED_TF32','RVC_HUBERT_FRONTEND_GEMM','RVC_NVTX')
$before = @{}
foreach ($key in $managed) { $before[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$oldPath = $env:PATH
$rows = [System.Collections.Generic.List[object]]::new()
$checks = [System.Collections.Generic.List[object]]::new()
$failures = [System.Collections.Generic.List[string]]::new()
function SetMode([string]$Mode, [bool]$Trace) {
    foreach ($key in $managed) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    $env:RVC_NSF_IMPLICIT_GEMM = '1'
    $env:RVC_RMVPE_GRU_PERSISTENT = '1'
    if ($Mode -in @('coalesced','gate_parallel','split2','split4')) { $env:RVC_RMVPE_GRU_WEIGHT_COALESCED = '1' }
    if ($Mode -eq 'gate_parallel') { $env:RVC_RMVPE_GRU_GATE_PARALLEL = '1' }
    if ($Mode -eq 'split2') { $env:RVC_RMVPE_GRU_COLUMN_SPLIT = '2' }
    if ($Mode -eq 'split4') { $env:RVC_RMVPE_GRU_COLUMN_SPLIT = '4' }
    $env:RVC_NVTX = if ($Trace) { '1' } else { '0' }
}
function InvokeCaptured([string]$Command, [string[]]$Arguments, [string]$Prefix) {
    Write-Host "Stage131 $Prefix"
    $stdout = Join-Path $work "$Prefix.stdout.log"
    $stderr = Join-Path $work "$Prefix.stderr.log"
    $argumentString = ($Arguments | ForEach-Object { if ($_ -match '[\s"]') { '"' + $_.Replace('"','\"') + '"' } else { $_ } }) -join ' '
    $proc = Start-Process -FilePath $Command -ArgumentList $argumentString -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($proc.ExitCode -ne 0) {
        $failures.Add("$Prefix exited $($proc.ExitCode)")
        Get-Content -LiteralPath $stderr -Tail 15 -ErrorAction SilentlyContinue | Write-Host
    }
    return $proc.ExitCode
}
function ArgsFor([string]$Output, [int]$Iterations, [bool]$Short) {
    if ($Short) {
        return [string[]]@('infer-wav-segment-benchmark',$voicePath,$hubertPath,$rmvpePath,$indexPath,
            $wavPath,$Output,'0','3',$IndexRate.ToString($inv),'0','cuda',"$Iterations")
    }
    return [string[]]@('infer-wav-full-benchmark',$voicePath,$hubertPath,$rmvpePath,$indexPath,
        $wavPath,$Output,$IndexRate.ToString($inv),'cuda',"$Iterations","$CoreFrames","$TensorBatchSize")
}
try {
    for ($cycle = 1; $cycle -le $Cycles; $cycle++) {
        $order = if ($cycle % 2 -eq 1) { @('control','coalesced','gate_parallel','split2','split4') } else { @('split4','split2','gate_parallel','coalesced','control') }
        foreach ($mode in $order) {
            SetMode $mode $false
            $name = "cycle${cycle}_$mode"
            $wav = Join-Path $work "$name.wav"
            if ((InvokeCaptured $exe (ArgsFor $wav $Runs $false) $name) -ne 0) { continue }
            $json = Get-Content -LiteralPath "$wav.benchmark.json" -Raw | ConvertFrom-Json
            $expected = if ($mode -ne 'control') { 'coalesced' } else { 'original' }
            $expectedSplit = if ($mode -eq 'split2') { 2 } elseif ($mode -eq 'split4') { 4 } else { 1 }
            if ($json.rmvpe_gru_execution -ne 'persistent' -or
                $json.rmvpe_gru_hidden_weight_layout -ne $expected -or
                [int]$json.rmvpe_gru_column_split -ne $expectedSplit -or
                [bool]$json.rmvpe_gru_gate_parallel -ne ($mode -eq 'gate_parallel') -or
                $json.nsf_conv_execution -ne 'implicit_gemm' -or
                $json.inference_ms.Count -ne $Runs -or
                $json.warm_median_ms -le 0 -or
                $json.iterations_exact_f32 -ne $true) {
                $failures.Add("$name failed route audit")
                continue
            }
            $rows.Add([pscustomobject]@{
                cycle=$cycle; mode=$mode; warm_median_ms=[double]$json.warm_median_ms
                first_ms=[double]$json.first_inference_ms; output_samples=[int]$json.output_samples
                sha256=(Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash; wav_path=$wav
            })
        }
    }
    $pcmHelper = Join-Path $PSScriptRoot 'Compare-Stage131-PCM.py'
    foreach ($cycle in 1..$Cycles) {
        $baseline = @($rows | Where-Object { $_.cycle -eq $cycle -and $_.mode -eq 'control' })
        if ($baseline.Count -ne 1) { $failures.Add("cycle$cycle missing control WAV"); continue }
        foreach ($mode in @('coalesced','gate_parallel','split2','split4')) {
            $candidate = @($rows | Where-Object { $_.cycle -eq $cycle -and $_.mode -eq $mode })
            if ($candidate.Count -ne 1) { $failures.Add("cycle$cycle missing $mode WAV"); continue }
            $tag = "cycle${cycle}_${mode}_pcm"
            if ((InvokeCaptured $PythonExe ([string[]]@($pcmHelper,$baseline[0].wav_path,$candidate[0].wav_path)) $tag) -ne 0) { continue }
            $metrics = Get-Content -LiteralPath (Join-Path $work "$tag.stdout.log") -Raw | ConvertFrom-Json
            $checks.Add([pscustomobject]@{ cycle=$cycle; mode=$mode; compatible=$metrics.compatible
                pcm16_bit_exact=$metrics.bit_exact; different_samples=$metrics.different_samples
                max_pcm_lsb=$metrics.max_pcm_lsb; snr_db=$metrics.snr_db })
            if (-not $metrics.compatible) { $failures.Add("$tag incompatible WAV format") }
            elseif ($mode -in @('coalesced','gate_parallel') -and -not $metrics.bit_exact) { $failures.Add("$tag violates bit-exact FP32 gate") }
            elseif ($mode -in @('split2','split4') -and $metrics.snr_db -lt $MinSnrDb) { $failures.Add("$tag SNR below $MinSnrDb dB") }
        }
    }
    if ($Profiler -ne 'measure') {
        if (-not (Test-Path -LiteralPath $NvtxDllDir -PathType Container)) { throw "Missing NVTX DLL directory $NvtxDllDir" }
        $env:PATH = "$NvtxDllDir;$oldPath"
        foreach ($mode in @('control','coalesced','gate_parallel','split2','split4')) {
            if (-not @($rows | Where-Object mode -EQ $mode)) { continue }
            SetMode $mode $true
            if ($Profiler -in @('nsys','all')) {
                $stem = Join-Path $work "$mode.nsys"
                $wav = Join-Path $work "$mode.nsys.wav"
                $args = [string[]]@('profile','--trace=cuda,nvtx','--sample=none','--force-overwrite=true','--output',$stem,$exe)
                $args += ArgsFor $wav 2 $false
                if ((InvokeCaptured (FullPath $NsysExe) $args "$mode.nsys") -eq 0) {
                    [void](InvokeCaptured (FullPath $NsysExe) ([string[]]@('export','--type','sqlite','--force-overwrite=true','--output',"$stem.sqlite","$stem.nsys-rep")) "$mode.nsys-export")
                    [void](InvokeCaptured (FullPath $NsysExe) ([string[]]@('stats','--report','cuda_gpu_kern_sum','--format','csv','--output',"$stem.summary","$stem.nsys-rep")) "$mode.nsys-summary")
                }
            }
            if ($Profiler -in @('ncu','all')) {
                $stem = Join-Path $work "$mode.bigru"
                $args = [string[]]@('--target-processes','all','--kernel-name','regex:bigru_(split_|gate_parallel_)?kernel',
                    '--launch-skip','0','--launch-count','4','--set','full','--force-overwrite',
                    '--export',"$stem.ncu-rep",$exe)
                $args += ArgsFor "$stem.wav" 2 $true
                if ((InvokeCaptured (FullPath $NcuExe) $args "$mode.bigru.ncu") -eq 0) {
                    $metrics = "$stem.ncu.csv"
                    $raw = "$stem.ncu-rep"
                    if (-not (Test-Path -LiteralPath $raw)) {
                        $failures.Add("$mode NCU report missing: $raw")
                    } else {
                        [void](InvokeCaptured (FullPath $NcuExe) ([string[]]@('--import',$raw,'--page','raw','--csv')) "$mode.bigru.ncu-export")
                        $rawCsv = Join-Path $work "$mode.bigru.ncu-export.stdout.log"
                        if (Test-Path -LiteralPath $rawCsv) { Copy-Item -LiteralPath $rawCsv -Destination $metrics -Force }
                    }
                }
            }
        }
        $analyzer = Join-Path $PSScriptRoot 'Analyze-Stage131-GRU-Nsight.py'
        foreach ($mode in @('control','coalesced','gate_parallel','split2','split4')) {
            $block = if ($mode -eq 'gate_parallel') { 768 } elseif ($mode -eq 'split2') { 512 } elseif ($mode -eq 'split4') { 1024 } else { 256 }
            $anArgs = [System.Collections.Generic.List[string]]::new()
            $anArgs.Add($analyzer)
            $anArgs.Add('--block-size'); $anArgs.Add("$block")
            $anArgs.Add('--kernel'); $anArgs.Add($(if ($mode -eq 'gate_parallel') { 'bigru_gate_parallel_kernel' } elseif ($mode -in @('split2','split4')) { 'bigru_split_kernel' } else { 'bigru_kernel' }))
            if ($Profiler -in @('nsys','all') -and (Test-Path -LiteralPath (Join-Path $work "$mode.nsys.sqlite"))) {
                $anArgs.Add('--nsys'); $anArgs.Add((Join-Path $work "$mode.nsys.sqlite"))
            }
            if ($Profiler -in @('ncu','all') -and (Test-Path -LiteralPath (Join-Path $work "$mode.bigru.ncu.csv"))) {
                $anArgs.Add('--ncu'); $anArgs.Add((Join-Path $work "$mode.bigru.ncu.csv"))
            }
            if ($anArgs.Count -gt 5) {
                $anArgs.Add('--output'); $anArgs.Add((Join-Path $work "$mode.gru-analysis.json"))
                [void](InvokeCaptured $PythonExe ($anArgs.ToArray()) "$mode.analyze")
            }
        }
    }
} finally {
    $manifest = [pscustomobject]@{
        stage=131; min_snr_db=$MinSnrDb; source_sha256=(Get-FileHash -LiteralPath $wavPath -Algorithm SHA256).Hash
        exe_sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash
        runs=$Runs; cycles=$Cycles; core_frames=$CoreFrames; tensor_batch_size=$TensorBatchSize
        profiler=$Profiler; benchmark=@($rows); pcm_checks=@($checks); failures=@($failures)
        note='Warm median excludes first execution. NSYS and NCU are separate processes; NCU uses replay and is not warm latency.'
    }
    $manifest | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $work 'manifest.json') -Encoding UTF8
    $rows | Export-Csv -LiteralPath (Join-Path $work 'summary.csv') -Encoding UTF8 -NoTypeInformation
    foreach ($key in $managed) { [Environment]::SetEnvironmentVariable($key,$before[$key],'Process') }
    $env:PATH = $oldPath
    if (-not $KeepWav) { Get-ChildItem -LiteralPath $work -Filter '*.wav' -File | Remove-Item -Force }
    $archive = Join-Path $root "$stamp.zip"
    $files = Get-ChildItem -LiteralPath $work -File | Where-Object { $_.Extension -notin @('.wav','.pth','.pt','.index','.exe','.dll') }
    Compress-Archive -LiteralPath @($files | ForEach-Object FullName) -DestinationPath $archive -Force
    Write-Host "Stage131 diagnostic ZIP: $archive"
}
if ($failures.Count) { throw "Stage131 FAILED: $($failures.Count) errors. Evidence retained." }
$rows | Format-Table cycle,mode,warm_median_ms,first_ms -AutoSize
Write-Host 'Stage131 A/B PASS; use NSYS/NCU results to judge GRU speedup.'
