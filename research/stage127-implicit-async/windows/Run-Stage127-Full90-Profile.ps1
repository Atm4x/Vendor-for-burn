param(
    [ValidateRange(2,16)][int]$Runs = 3,
    [ValidateRange(1,6)][int]$Cycles = 2,
    [string]$Modes = 'implicit_sync,implicit_async,hybrid_sync,ntc_combined',
    [ValidateSet('none','nsys','ncu','all')][string]$Profiler = 'all',
    [string]$ProfileModes = 'implicit_sync,implicit_async,ntc_combined',
    [ValidateSet('fp16','fp32','both')][string]$TorchPrecision = 'both',
    [string]$PythonExe = 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe',
    [string]$OriginalRoot = 'D:\Projects\tts-with-rvc',
    [string]$RustExe = '.\tts-with-rvc-rs.exe',
    [string]$AssetsDir = '.\embed',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [string]$OutDir = '.\stage127_full90_results',
    [ValidateRange(25,900)][int]$CoreFrames = 300,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(0,1)][double]$IndexRate = 0.75,
    [string]$NcuExe = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe',
    [string]$NsysExe = 'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe',
    [string]$NvtxDllDir = 'C:\Users\mrtak\miniconda3\envs\torch2.6.0\Lib\site-packages\torch\lib',
    [string]$NcuKernelRegex = '',
    [ValidateRange(0,100000)][int]$NcuLaunchSkip = 0,
    [ValidateRange(1,20)][int]$NcuLaunchCount = 4,
    [ValidateRange(0,10000)][double]$NcuStartSeconds = 0,
    [ValidateRange(0.2,9)][double]$NcuDurationSeconds = 3,
    [switch]$TorchNsys,
    [switch]$SkipTorch,
    [switch]$CheckOnly,
    [switch]$KeepWav
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
function Absolute([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $root $Path))
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Missing configured Python: $PythonExe" }
$runner = Join-Path $root 'Run-Stage127-Full90-Profile.py'
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Missing Stage127 Python runner: $runner" }
$inv = [Globalization.CultureInfo]::InvariantCulture
$arguments = @(
    $runner, '--torch-python', (Absolute $PythonExe), '--original-root', (Absolute $OriginalRoot),
    '--rust-exe', (Absolute $RustExe), '--assets-dir', (Absolute $AssetsDir),
    '--input-wav', (Absolute $InputWav), '--out-dir', (Absolute $OutDir),
    '--runs', "$Runs", '--cycles', "$Cycles", '--modes', $Modes,
    '--profiler', $Profiler, '--profile-modes', $ProfileModes,
    '--torch-precision', $TorchPrecision, '--core-frames', "$CoreFrames",
    '--batch-size', "$TensorBatchSize", '--index-rate', $IndexRate.ToString($inv),
    '--ncu-exe', (Absolute $NcuExe), '--nsys-exe', (Absolute $NsysExe),
    '--nvtx-dll-dir', (Absolute $NvtxDllDir),
    '--ncu-launch-skip', "$NcuLaunchSkip", '--ncu-launch-count', "$NcuLaunchCount",
    '--ncu-start-seconds', $NcuStartSeconds.ToString($inv),
    '--ncu-duration-seconds', $NcuDurationSeconds.ToString($inv)
)
if ($NcuKernelRegex) { $arguments += @('--ncu-kernel-regex', $NcuKernelRegex) }
if ($TorchNsys) { $arguments += '--torch-nsys' }
if ($SkipTorch) { $arguments += '--skip-torch' }
if ($CheckOnly) { $arguments += '--check-only' }
if ($KeepWav) { $arguments += '--keep-wav' }
Write-Host "Stage127 full 90s: $Modes; profiler=$Profiler; runs=$Runs cycles=$Cycles"
& $PythonExe @arguments
if ($LASTEXITCODE -ne 0) { throw "Stage127 failed ($LASTEXITCODE). Diagnostic ZIP is in $OutDir" }
