param(
    [ValidateRange(2,16)][int]$Runs=3,
    [ValidateRange(1,6)][int]$Cycles=2,
    [string]$Modes='control,linear,residue,combined',
    [ValidateSet('both','fp32','fp16')][string]$TorchPrecision='both',
    [string]$PythonExe='D:\Projects\tts-with-rvc\.venv\Scripts\python.exe',
    [string]$OriginalRoot='D:\Projects\tts-with-rvc',
    [string]$RustExe='.\tts-with-rvc-rs.exe',
    [string]$AssetsDir='.\embed',
    [string]$InputWav='.\embed\MitaKind.wav',
    [string]$OutDir='.\stage126_full90_results',
    [ValidateRange(25,900)][int]$CoreFrames=300,
    [ValidateRange(1,32)][int]$TensorBatchSize=2,
    [ValidateRange(0,1)][double]$IndexRate=0.75,
    [switch]$CheckOnly,[switch]$SkipTorch,[switch]$KeepWav
)
$ErrorActionPreference='Stop'
function Absolute([string]$value){
    if([IO.Path]::IsPathRooted($value)){return [IO.Path]::GetFullPath($value)}
    return [IO.Path]::GetFullPath((Join-Path $PSScriptRoot $value))
}
if(-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)){throw "Missing the configured uv Python: $PythonExe"}
$script=Join-Path $PSScriptRoot 'Run-Stage126-Full90-Compare.py'
if(-not (Test-Path -LiteralPath $script -PathType Leaf)){throw "Missing packaged Stage126 Python driver: $script"}
$invariant=[Globalization.CultureInfo]::InvariantCulture
$arguments=@($script,'--torch-python',(Absolute $PythonExe),'--original-root',(Absolute $OriginalRoot),'--rust-exe',(Absolute $RustExe),'--assets-dir',(Absolute $AssetsDir),'--input-wav',(Absolute $InputWav),'--runs',"$Runs",'--cycles',"$Cycles",'--modes',$Modes,'--torch-precision',$TorchPrecision,'--core-frames',"$CoreFrames",'--batch-size',"$TensorBatchSize",'--index-rate',$IndexRate.ToString($invariant),'--out-dir',(Absolute $OutDir))
if($CheckOnly){$arguments+='--check-only'}
if($SkipTorch){$arguments+='--skip-torch'}
if($KeepWav){$arguments+='--keep-wav'}
Write-Host "Stage126 full 90s: Rust $Modes; PyTorch $TorchPrecision; cycles=$Cycles; runs=$Runs"
& $PythonExe @arguments
if($LASTEXITCODE -ne 0){throw "Stage126 run failed (exit $LASTEXITCODE). Diagnostic ZIP in $OutDir"}
