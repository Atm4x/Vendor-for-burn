param(
    [ValidateSet('cuda')][string]$Device = 'cuda',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(2,16)][int]$WarmRuns = 2,
    [ValidateRange(25,900)][int]$CoreFrames = 300,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(0,1)][double]$IndexRate = 0.75,
    [ValidateSet('all','baseline','gemm','tf32x2','batched')][string]$Modes = 'all',
    [ValidateSet('both','fp16','fp32')][string]$TorchPrecision = 'both',
    [string]$TorchPython = 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe',
    [string]$OriginalRoot = '',
    [string]$Grid = ''
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$assets = Join-Path $root 'embed'
$exe = Join-Path $root 'tts-with-rvc-rs.exe'
if ([string]::IsNullOrWhiteSpace($OriginalRoot)) { $OriginalRoot = Join-Path $root 'reference_upstream' }
$driver = Join-Path $root 'Run-RVC-Stage80-Compare.py'

function Get-EmbedFileName([string]$requested, [string]$description) {
    $value = $requested.Trim()
    if ([IO.Path]::IsPathRooted($value)) {
        $absolute = [IO.Path]::GetFullPath($value)
    } elseif ($value -match '^(\.\\|\./)?embed[\\/]') {
        $relative = $value -replace '^(\.\\|\./)?', ''
        $absolute = [IO.Path]::GetFullPath((Join-Path $root $relative))
    } else {
        $absolute = [IO.Path]::GetFullPath((Join-Path $assets $value))
    }
    if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
        throw "Missing $description file: $absolute"
    }
    $parent = [IO.Path]::GetFullPath($assets).TrimEnd('\', '/')
    if (-not $absolute.StartsWith(($parent + [IO.Path]::DirectorySeparatorChar), [StringComparison]::OrdinalIgnoreCase)) {
        throw "$description must be in the existing embed folder: $absolute"
    }
    return $absolute.Substring($parent.Length + 1)
}

foreach ($item in @($TorchPython,$exe,$driver)) {
    if (-not (Test-Path -LiteralPath $item -PathType Leaf)) { throw "Missing runtime: $item" }
}
if (-not (Test-Path -LiteralPath (Join-Path $OriginalRoot 'tts_with_rvc\infer\vc\pipeline.py') -PathType Leaf)) {
    throw "Missing original Python RVC sources in $OriginalRoot. Use -OriginalRoot to set their location."
}
if ($CoreFrames + 84 -gt 1000) {
    throw 'Invalid core: core frames + 84 must not exceed 1000'
}

$voiceName = Get-EmbedFileName $Voice 'Voice'
$hubertName = Get-EmbedFileName $Hubert 'HuBERT'
$rmvpeName = Get-EmbedFileName $Rmvpe 'RMVPE'
$inputName = Get-EmbedFileName $InputWav 'InputWav'
$indexName = if ($IndexRate -gt 0) { Get-EmbedFileName $Index 'Index' } else { 'CrazyMita.index' }

$nativeModes = switch ($Modes) {
    'all'      { 'direct,gemm,tf32x2' }
    'baseline' { 'direct' }
    'gemm'     { 'direct,gemm' }
    'tf32x2'   { 'direct,tf32x2' }
    'batched'  { 'direct,batched' }
}
if ([string]::IsNullOrWhiteSpace($Grid)) {
    $candidates = @("${CoreFrames}:${TensorBatchSize}", '600:1', '900:1')
    $Grid = (@($candidates | Select-Object -Unique) -join ',')
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$output = Join-Path $root "stage80_original_comparison_$stamp"
$invariant = [Globalization.CultureInfo]::InvariantCulture

Write-Host "RVC Stage80: device=$Device, grid=$Grid, modes=$nativeModes, torch=$TorchPrecision"
Write-Host "Pinned original source snapshot: $OriginalRoot"
Write-Host "Using existing embed: $assets"
Write-Host "Using original PyTorch: $TorchPython"
& $TorchPython $driver `
    --torch-python $TorchPython `
    --original-root $OriginalRoot `
    --native-exe $exe `
    --assets-dir $assets `
    --voice $voiceName --hubert $hubertName --rmvpe $rmvpeName `
    --index $indexName --input-wav $inputName `
    --index-rate $IndexRate.ToString($invariant) --runs $WarmRuns `
    --grid $Grid --native-modes $nativeModes `
    --torch-precision $TorchPrecision --out $output
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Stage80 benchmark failed (exit $code). Check $output for logs." }
Write-Host "Result: $(Join-Path $root 'RVC_STAGE79_ORIGINAL_COMPARISON_RESULTS.zip')"
