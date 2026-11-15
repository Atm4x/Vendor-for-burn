param(
    [string]$RustExe = '.\tts-with-rvc-rs-stage102.exe',
    [string]$PythonExe = 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe',
    [string]$OriginalRoot = 'D:\Projects\tts-with-rvc',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(0,1)][double]$IndexRate = 0.75,
    [ValidateSet('both','rust','torch')][string]$Implementation = 'both',
    [ValidateSet('both','nsys','ncu')][string]$Profiler = 'both',
    [ValidateSet('fp16','fp32')][string]$TorchPrecision = 'fp16',
    [ValidateRange(2,6)][int]$Runs = 2,
    [ValidateRange(1,916)][int]$CoreFrames = 300,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(1,8)][int]$NcuLaunchCount = 3,
    [ValidateRange(0,1000)][int]$NcuLaunchSkip = 0,
    [string]$NsysExe = 'nsys',
    [string]$NcuExe = 'ncu'
)
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture
$root = Join-Path $PSScriptRoot 'stage102_nsight_work'
$reports = Join-Path $root 'reports'
$scratch = Join-Path $root 'scratch'
New-Item -ItemType Directory -Path $reports,$scratch -Force | Out-Null
$torchScript = Join-Path $PSScriptRoot 'torch_stage102_profile.py'
$required = @($PythonExe, $InputWav, $Voice, $Hubert, $Rmvpe)
if ($IndexRate -gt 0) { $required += $Index }
if ($Implementation -ne 'torch') { $required += $RustExe }
if ($Implementation -ne 'rust') { $required += $torchScript; $required += (Join-Path $OriginalRoot 'tts_with_rvc/infer/vc/pipeline.py') }
foreach ($item in $required) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw "Missing input: $item" }
}
if ($CoreFrames + 84 -gt 1000) { throw 'CoreFrames + 84 must be <=1000' }
if ($Profiler -ne 'ncu' -and -not (Get-Command $NsysExe -ErrorAction SilentlyContinue)) { throw 'Nsight Systems nsys not found' }
if ($Profiler -ne 'nsys' -and -not (Get-Command $NcuExe -ErrorAction SilentlyContinue)) { throw 'Nsight Compute ncu not found' }
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$wav = (Resolve-Path -LiteralPath $InputWav).Path
$voicePath = (Resolve-Path -LiteralPath $Voice).Path
$hubertPath = (Resolve-Path -LiteralPath $Hubert).Path
$rmvpePath = (Resolve-Path -LiteralPath $Rmvpe).Path
$indexPath = if ($IndexRate -eq 0) { '-' } else { (Resolve-Path -LiteralPath $Index).Path }
$rust = if ($Implementation -eq 'torch') { $null } else { (Resolve-Path -LiteralPath $RustExe).Path }
$variants = if ($Implementation -eq 'both') { @('rust','torch') } else { @($Implementation) }
$envNames = @('RVC_NVTX','RVC_NSF_IMPLICIT_GEMM','RVC_RMVPE_GRU_PERSISTENT','RVC_GLOBAL_F0',
    'RVC_NSF_GLOBAL_PHASE','RVC_NSF_GLOBAL_NOISE','RVC_BATCH_SCHEDULE','RVC_CHUNK_STRATEGY',
    'RVC_RMVPE_BN_FOLD','RVC_RMVPE_UNET_WINOGRAD','RVC_RMVPE_UNET_SPATIAL_QUAD',
    'RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32','RVC_RMVPE_UNET_COMPENSATED_TF32',
    'RVC_RMVPE_UNET_GEMM','RVC_RMVPE_UNET_CONV2D_GEMM','RVC_RMVPE_UNET_TRANSPOSE2D_GEMM',
    'RVC_HUBERT_FRONTEND_GEMM','RVC_GPU_FAISS','RVC_GPU_F0','RVC_DIAGNOSTIC_TRACE','RVC_PROFILE')
$oldEnv = @{}
foreach ($name in $envNames) { $oldEnv[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
function Invoke-LoggedNative([string]$Tool, [string[]]$Arguments, [string]$LogPath) {
    $original = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Tool @Arguments *>&1 | Out-File -FilePath $LogPath -Encoding utf8
        $exit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $original }
    if ($exit -ne 0) {
        Get-Content -LiteralPath $LogPath -Tail 35 | ForEach-Object { Write-Host $_ }
        throw "Profiler failed with exit $exit : $LogPath"
    }
}
try {
    foreach ($name in $envNames) { [Environment]::SetEnvironmentVariable($name,$null,'Process') }
    $env:RVC_NVTX = '1'
    $env:RVC_NSF_IMPLICIT_GEMM = '1'
    $env:RVC_RMVPE_GRU_PERSISTENT = '1'
    $env:RVC_GLOBAL_F0 = '1'
    $env:RVC_NSF_GLOBAL_PHASE = '1'
    $env:RVC_NSF_GLOBAL_NOISE = '1'
    $env:RVC_CHUNK_STRATEGY = 'cost-aware'
    $env:RVC_BATCH_SCHEDULE = 'contiguous'
    if ($Implementation -ne 'torch') {
        $nvtxStatus = & $rust nvtx-status
        if ($LASTEXITCODE -ne 0) { throw 'Stage102 Rust executable does not support nvtx-status' }
        if ($nvtxStatus -ne 'nvtx_available=true') {
            throw 'Rust NVTX unavailable. Put nvToolsExt64_1.dll in PATH (CUDA Toolkit / NVIDIA NVTX installation); do not profile unmarked warmup kernels.'
        }
    }
    $hashes = [ordered]@{}
    foreach ($f in @($Voice,$Hubert,$Rmvpe,$InputWav) + $(if ($IndexRate -gt 0) {@($Index)} else {@()})) {
        $hashes[(Split-Path $f -Leaf)] = (Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $manifest = [ordered]@{
        stage=102; source_baseline_commit='d8a0f22f3a47fd7ab8ac570791bd2167bf3b322e'
        utc=(Get-Date).ToUniversalTime().ToString('o'); implementation=$Implementation; profiler=$Profiler
        runs=$Runs; core_frames=$CoreFrames; tensor_batch_size=$TensorBatchSize
        precision_torch=$TorchPrecision; precision_rust='fp32'; index_rate=$IndexRate
        ncu_launch_count=$NcuLaunchCount; ncu_launch_skip=$NcuLaunchSkip
        filenames_sha256=$hashes
        limitations=@('Different Torch vs Rust preprocessing and chunk geometry',
            'Torch FP16 versus Rust FP32 are not precision-matched',
            'NSYS/NCU instrumented timings are not uninstrumented inference latency')
        output_audio_in_zip=$false
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $reports 'manifest.json') -Encoding UTF8
    & $python -c 'import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name())' 2>&1 |
        Out-File -LiteralPath (Join-Path $reports 'pytorch_cuda.txt') -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch CUDA preflight failed' }
    if ($Profiler -ne 'ncu') {
        & $NsysExe --version | Out-File (Join-Path $reports 'nsys-version.txt') -Encoding utf8
    }
    if ($Profiler -ne 'nsys') {
        & $NcuExe --version | Out-File (Join-Path $reports 'ncu-version.txt') -Encoding utf8
    }
    foreach ($impl in $variants) {
        $profileOut = Join-Path $scratch "$impl-output.wav"
        if ($impl -eq 'rust') {
            $command = $rust
            $arguments = @('infer-wav-full-benchmark',$voicePath,$hubertPath,$rmvpePath,$indexPath,
                $wav,$profileOut,$IndexRate.ToString($inv),'cuda',"$Runs","$CoreFrames","$TensorBatchSize")
            $kernelRegex = 'direct_conv2d_kernel'
            $nvtxLabel = 'stage102_rust_measured/'
        } else {
            $command = $python
            $arguments = @($torchScript,'--original-root',(Resolve-Path -LiteralPath $OriginalRoot).Path,
                '--assets-dir',(Resolve-Path -LiteralPath (Split-Path $Voice -Parent)).Path,
                '--voice',(Split-Path $Voice -Leaf),'--hubert',(Split-Path $Hubert -Leaf),
                '--rmvpe',(Split-Path $Rmvpe -Leaf),'--index',(Split-Path $Index -Leaf),
                '--input-wav',(Split-Path $InputWav -Leaf),'--index-rate',$IndexRate.ToString($inv),
                '--precision',$TorchPrecision,'--runs',"$Runs",'--output',(Join-Path $reports 'torch-summary'))
            $kernelRegex = 'cutlass|sm80_xmma|cudnn::cnn::conv2d'
            $nvtxLabel = 'stage102_torch_measured/'
        }
        if ($Profiler -ne 'ncu') {
            $nsysOutput = Join-Path $reports "$impl-nsys"
            $nsysArguments = @('profile','--trace=cuda,nvtx','--sample=none','--force-overwrite=true',
                "--output=$nsysOutput",$command) + $arguments
            Write-Host "Stage102 Nsight Systems: $impl"
            Invoke-LoggedNative $NsysExe $nsysArguments (Join-Path $reports "$impl-nsys.log")
        }
        if ($Profiler -ne 'nsys') {
            $ncuOutput = Join-Path $reports "$impl-conv2d"
            $ncuArguments = @('--target-processes','application-only','--nvtx','--nvtx-include',$nvtxLabel,
                '--kernel-name',"regex:$kernelRegex",'--launch-skip',"$NcuLaunchSkip",
                '--launch-count',"$NcuLaunchCount",'--section','SpeedOfLight',
                '--section','LaunchStats','--section','Occupancy','--section','MemoryWorkloadAnalysis',
                '--section','ComputeWorkloadAnalysis','--export',$ncuOutput,'--force-overwrite',$command) + $arguments
            Write-Host "Stage102 Nsight Compute: $impl; selected kernels only"
            Invoke-LoggedNative $NcuExe $ncuArguments (Join-Path $reports "$impl-ncu.log")
            $ncuReport = "$ncuOutput.ncu-rep"
            if (-not (Test-Path -LiteralPath $ncuReport)) { throw "Missing Nsight Compute report: $ncuReport" }
            $csv = Join-Path $reports "$impl-conv2d.csv"
            Invoke-LoggedNative $NcuExe @('--import',$ncuReport,'--page','raw','--csv') $csv
            if (-not ((Get-Content -LiteralPath $csv | Measure-Object -Line).Lines -gt 5)) {
                throw "No NCU metrics exported for $impl; check kernel-name filter and NVTX ranges"
            }
        }
        if ($impl -eq 'rust') {
            $benchmarkJson = "$profileOut.benchmark.json"
            if (-not (Test-Path -LiteralPath $benchmarkJson)) { throw 'Rust benchmark JSON missing' }
            Copy-Item -LiteralPath $benchmarkJson -Destination (Join-Path $reports 'rust-benchmark.json')
        }
        Remove-Item -LiteralPath $profileOut -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath "$profileOut.benchmark.json" -Force -ErrorAction SilentlyContinue
    }
} finally {
    foreach ($name in $oldEnv.Keys) { [Environment]::SetEnvironmentVariable($name,$oldEnv[$name],'Process') }
    $archive = Join-Path $PSScriptRoot 'RVC_STAGE102_NSYS_NCU_RESULTS.zip'
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    if (Test-Path -LiteralPath $reports) {
        Compress-Archive -Path (Join-Path $reports '*') -DestinationPath $archive -CompressionLevel Optimal
        Write-Host "Stage102 profile results (no WAV, no model weights): $archive"
    }
}
