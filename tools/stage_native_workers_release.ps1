param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{40}$')][string]$Revision,
    [Parameter(Mandatory)][string]$InputRoot,
    [Parameter(Mandatory)][string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$families = @('beat-this', 'dettato', 'f5-tts', 'fish-speech-1.5', 'gigaam-v3', 'rvc-v2', 'silero-tts', 'omnivoice', 'whisper', 'semantic')
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new release staging directory' }
$packages = @(Get-ChildItem -LiteralPath $InputRoot -File -Filter '*.zip' -Recurse)
if ($packages.Count -ne $families.Count) {
    throw "Expected $($families.Count) package ZIPs, found $($packages.Count)"
}

$entries = foreach ($family in $families) {
    $name = "${family}_v${Version}_win_x86_64.zip"
    $matches = @($packages | Where-Object Name -CEQ $name)
    if ($matches.Count -ne 1) { throw "Expected exactly one $name" }
    $zip = $matches[0]
    $archive = [IO.Compression.ZipFile]::OpenRead($zip.FullName)
    try {
        $manifests = @($archive.Entries | Where-Object FullName -CIn @('worker-package.json', 'worker-manifest.json'))
        if ($manifests.Count -ne 1) { throw "Expected one worker manifest in $name" }
        $reader = [IO.StreamReader]::new($manifests[0].Open())
        try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        if ($manifest.package_id -cne $family -or $manifest.package_version -cne $Version -or
            $manifest.source_revision -cne $Revision -or $manifest.os -cne 'windows' -or
            $manifest.arch -cne 'x86_64') {
            throw "Mismatched identity, source revision or platform in $name"
        }
        $executables = @($archive.Entries | Where-Object { $_.Name -like '*.exe' })
        if ($executables.Count -ne 1 -or $executables[0].FullName -cne $manifest.executable.path -or
            $executables[0].Length -ne $manifest.executable.size_bytes) {
            throw "Worker executable does not match the manifest in $name"
        }
        $stream = $executables[0].Open()
        try {
            $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant()
        } finally { $stream.Dispose() }
        if ($hash -cne $manifest.executable.sha256) { throw "Worker executable SHA-256 does not match in $name" }
    } finally {
        $archive.Dispose()
    }
    [ordered]@{
        package_id = $family
        path = $name
        size_bytes = $zip.Length
        sha256 = (Get-FileHash -LiteralPath $zip.FullName).Hash.ToLowerInvariant()
    }
}

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
foreach ($zip in $packages) { Copy-Item -LiteralPath $zip.FullName -Destination $OutputRoot }
[ordered]@{
    schema_version = 1
    package_version = $Version
    source_revision = $Revision
    os = 'windows'
    arch = 'x86_64'
    packages = @($entries)
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputRoot 'staged-release-index.json') -Encoding utf8NoBOM
