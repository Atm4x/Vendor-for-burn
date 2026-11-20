param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][ValidateSet('cuda', 'cpu', 'wgpu')][string]$Backend,
    [Parameter(Mandatory)][string]$ExePath,
    [Parameter(Mandatory)][string]$OutputRoot,
    [string]$Name = 'rvc_v2'
)

$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new empty package destination' }
$exe = Get-Item -LiteralPath $ExePath -ErrorAction Stop
if ($exe.PSIsContainer -or ($exe.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    throw 'Expected a regular worker executable'
}
if ($exe.Name -ne 'rvc-worker.exe') { throw 'Expected rvc-worker.exe' }
$root = Join-Path $OutputRoot 'package'
New-Item -ItemType Directory -Path $root -Force | Out-Null
Copy-Item -LiteralPath $exe.FullName -Destination (Join-Path $root 'rvc-worker.exe')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'IPC_STDIO_WORKER.md') -Destination $root
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'distribution.json') -Destination $root
$binary = Join-Path $root 'rvc-worker.exe'
$manifest = [ordered]@{
    schema_version = 1
    worker_protocol = 1
    name = $Name
    version = $Version
    backend = $Backend
    executable = [ordered]@{
        path = 'rvc-worker.exe'
        size_bytes = (Get-Item -LiteralPath $binary).Length
        sha256 = (Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    required_launch_arguments = @('--backend', '--voice', '--hubert', '--rmvpe', '--input-dir', '--output-dir')
    private_weights_included = $false
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $root 'worker-manifest.json') -Encoding utf8NoBOM
Write-Host 'Worker packaged without model assets; supply explicit weight paths at launch.'
