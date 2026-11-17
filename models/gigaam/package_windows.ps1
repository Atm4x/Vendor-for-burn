[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][string]$ModelId,
    [Parameter(Mandatory)][string]$CheckpointPath,
    [Parameter(Mandatory)][string]$ExecutablePath,
    [Parameter(Mandatory)][string]$OutputRoot,
    [string]$TokenizerPath,
    [ValidateSet('cuda','cpu')][string]$Backend = 'cuda',
    [string]$CheckpointUrl,
    [string]$TokenizerUrl
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$root = $PSScriptRoot
$definition = Get-Content -LiteralPath (Join-Path $root 'distribution.json') -Raw | ConvertFrom-Json
if (-not $ModelId.StartsWith($definition.model_id_prefix)) { throw 'Invalid GigaAM model ID' }
$name = $ModelId.Substring($definition.model_id_prefix.Length)
if ($name -notin $definition.supported_models) { throw 'Unknown GigaAM model ID' }
$versionMatch = Select-String -Path (Join-Path $root 'Cargo.toml') -Pattern '^version = "([0-9.]+)"$' | Select-Object -First 1
if ($versionMatch.Matches[0].Groups[1].Value -ne $Version) { throw 'Package version must match Cargo.toml' }
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new output directory' }
if (-not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) { throw 'Checkpoint file missing' }
if (-not (Test-Path -LiteralPath $ExecutablePath -PathType Leaf)) { throw 'Worker binary missing' }
if (($name -match 'e2e_|^v1_rnnt$') -and -not $TokenizerPath) { throw 'SentencePiece tokenizer required for this model' }
if ($TokenizerPath -and -not (Test-Path -LiteralPath $TokenizerPath -PathType Leaf)) { throw 'Tokenizer file missing' }
if ($TokenizerUrl -and -not $TokenizerPath) { throw 'Tokenizer URL requires a tokenizer file' }

function New-FileRef([string]$Path,[string]$Name) {
    return [ordered]@{
        path = $Name
        size_bytes = (Get-Item -LiteralPath $Path).Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$assetsDir = Join-Path $OutputRoot 'assets'
$packageDir = Join-Path $OutputRoot 'package'
$modelDir = Join-Path $packageDir 'model'
New-Item -ItemType Directory -Path $assetsDir,$modelDir -Force | Out-Null
$exe = Join-Path $modelDir 'gigaam-rs.exe'
Copy-Item -LiteralPath $ExecutablePath -Destination $exe

$checkpointName = "$name.ckpt"
$weights = @([ordered]@{ role = 'checkpoint'; file = (New-FileRef $CheckpointPath $checkpointName) })
if ($CheckpointUrl) { $weights[0]['url'] = $CheckpointUrl }
if ($TokenizerPath) {
    $tokenizerName = "${name}_tokenizer.model"
    $weights += [ordered]@{ role = 'tokenizer'; file = (New-FileRef $TokenizerPath $tokenizerName) }
    if ($TokenizerUrl) { $weights[-1]['url'] = $TokenizerUrl }
}

$variant = if ($Backend -eq 'cuda') { 'burn_cubecl_cuda_fp32' } else { 'burn_flex_cpu_fp32' }
$manifest = [ordered]@{
    schema_version = 1
    model_id = $ModelId
    display_name = "GigaAM $name"
    package_version = $Version
    os = 'windows'
    arch = 'x86_64'
    protocol_version = $definition.worker_protocol
    backends = @($Backend)
    executable = (New-FileRef $exe 'gigaam-rs.exe')
    assets = @($weights | ForEach-Object { $_.file })
    runtime_files = @()
}
$prefix = "$ModelId-v$Version-windows-x64-$Backend"
$manifestPath = Join-Path $assetsDir "$prefix.native-model.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding utf8NoBOM
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $modelDir 'native-model.json')
Copy-Item -LiteralPath (Join-Path $root 'LICENSE') -Destination (Join-Path $packageDir 'LICENSE.GigaAM')
Copy-Item -LiteralPath (Join-Path $root 'THIRD_PARTY_NOTICES.md') -Destination (Join-Path $packageDir 'THIRD_PARTY_NOTICES.md')
Copy-Item -LiteralPath (Join-Path $root 'WORKER_PROTOCOL.md') -Destination (Join-Path $packageDir 'WORKER_PROTOCOL.md')

$publishedExe = Join-Path $assetsDir "$prefix.exe"
Copy-Item -LiteralPath $exe -Destination $publishedExe
$zip = Join-Path $assetsDir "$prefix.zip"
Compress-Archive -Path (Join-Path $packageDir '*') -DestinationPath $zip -CompressionLevel Optimal
$sourceRevision = git -C (Join-Path $root '../..') rev-parse HEAD
$plan = [ordered]@{
    schema_version = 1
    model_id = $ModelId
    display_name = "GigaAM $name"
    version = $Version
    variant = $variant
    source_revision = $sourceRevision
    worker_protocol = $definition.worker_protocol
    gigaam_contract_version = $definition.gigaam_contract_version
    package = (New-FileRef $zip "$prefix.zip")
    binary = (New-FileRef $publishedExe "$prefix.exe")
    manifest = (New-FileRef $manifestPath "$prefix.native-model.json")
    weights = $weights
    runtime = @()
}
$plan | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $assetsDir "$prefix.install-plan.json") -Encoding utf8NoBOM
Get-ChildItem -LiteralPath $assetsDir -File | Sort-Object Name | ForEach-Object {
    "$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant())  $($_.Name)"
} | Set-Content -LiteralPath (Join-Path $assetsDir "$prefix.sha256") -Encoding ascii
