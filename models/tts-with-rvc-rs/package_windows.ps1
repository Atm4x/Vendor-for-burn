param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][ValidatePattern('^rvc_v2_[a-z0-9_]+$')][string]$ModelId,
    [Parameter(Mandatory)][ValidateSet('cuda', 'cpu', 'wgpu')][string]$Backend,
    [Parameter(Mandatory)][string]$ExePath,
    [Parameter(Mandatory)][string]$VoicePath,
    [Parameter(Mandatory)][string]$HubertPath,
    [Parameter(Mandatory)][string]$RmvpePath,
    [string]$IndexPath,
    [Parameter(Mandatory)][string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$project = Get-Content -Raw (Join-Path $PSScriptRoot 'distribution.json') | ConvertFrom-Json
if ($project.worker_protocol -ne 1 -or $project.rvc_contract_version -ne 1) {
    throw 'Unsupported worker distribution contract'
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw 'Use a new empty package destination'
}

function Assert-SourceFile([string]$Path) {
    $item = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Expected regular non-symlink source file: $Path"
    }
    return $item.FullName
}
function FileRef([string]$Path, [string]$Relative) {
    return [ordered]@{
        path = $Relative
        size_bytes = (Get-Item -LiteralPath $Path).Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$resolvedExe = Assert-SourceFile $ExePath
if ([IO.Path]::GetFileName($resolvedExe) -ne 'rvc-worker.exe') {
    throw 'Expected a compiled rvc-worker.exe binary'
}
$resolvedVoice = Assert-SourceFile $VoicePath
$resolvedHubert = Assert-SourceFile $HubertPath
$resolvedRmvpe = Assert-SourceFile $RmvpePath
$resolvedIndex = if ($IndexPath) { Assert-SourceFile $IndexPath } else { $null }

$stage = Join-Path $OutputRoot 'package'
$assets = Join-Path $stage 'model'
New-Item -ItemType Directory -Path $assets -Force | Out-Null
Copy-Item -LiteralPath $resolvedExe -Destination (Join-Path $stage 'rvc-worker.exe')
Copy-Item -LiteralPath $resolvedVoice -Destination (Join-Path $assets 'voice.pth')
Copy-Item -LiteralPath $resolvedHubert -Destination (Join-Path $assets 'hubert.pt')
Copy-Item -LiteralPath $resolvedRmvpe -Destination (Join-Path $assets 'rmvpe.pt')
if ($resolvedIndex) { Copy-Item -LiteralPath $resolvedIndex -Destination (Join-Path $assets 'index.index') }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'IPC_STDIO_WORKER.md') -Destination $stage
$variant = switch ($Backend) {
    'cuda' { 'burn_cubecl_cuda_fp32' }
    'wgpu' { 'burn_cubecl_wgpu_fp32' }
    'cpu' { 'burn_cubecl_cpu_fp32' }
}
$files = @(
    FileRef (Join-Path $assets 'voice.pth') 'model/voice.pth'
    FileRef (Join-Path $assets 'hubert.pt') 'model/hubert.pt'
    FileRef (Join-Path $assets 'rmvpe.pt') 'model/rmvpe.pt'
)
if ($resolvedIndex) { $files += FileRef (Join-Path $assets 'index.index') 'model/index.index' }
$manifest = [ordered]@{
    schema_version = 1
    model_id = $ModelId
    display_name = "RVC voice $ModelId"
    package_version = $Version
    os = 'windows'
    arch = 'x86_64'
    protocol_version = 1
    rvc_contract_version = 1
    service = 'voice_conversion'
    backends = @($Backend)
    execution_variant = $variant
    executable = FileRef (Join-Path $stage 'rvc-worker.exe') 'rvc-worker.exe'
    assets = $files
    runtime_files = @()
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $stage 'native-model.json') -Encoding utf8NoBOM
$source = git -C $root rev-parse HEAD
[ordered]@{
    schema_version = 1
    model_id = $ModelId
    source_revision = $source
    version = $Version
    variant = $variant
    worker_protocol = 1
    model_package = 'package'
    private_assets_included = $true
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $OutputRoot 'install-plan.json') -Encoding utf8NoBOM
Write-Host 'Package created for local installation only. It contains private model files; do NOT upload it to CI or GitHub.'
