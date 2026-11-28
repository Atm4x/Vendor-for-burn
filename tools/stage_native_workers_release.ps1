param(
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{40}$')][string]$Revision,
    [Parameter(Mandatory)][string]$Matrix,
    [Parameter(Mandatory)][string]$InputRoot,
    [Parameter(Mandatory)][string]$OutputRoot,
    [Parameter(Mandatory)][string]$ExistingIndex,
    [Parameter(Mandatory)][string]$ExistingAssets
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new release staging directory' }
$index = Get-Content -LiteralPath $ExistingIndex -Raw | ConvertFrom-Json
if ($index.schema_version -notin 1, 2 -or $index.os -cne 'windows' -or $index.arch -cne 'x86_64') {
    throw 'Unsupported Windows release index'
}
$assets = @{}
foreach ($asset in (Get-Content -LiteralPath $ExistingAssets -Raw | ConvertFrom-Json)) {
    $assets.Add($asset.name, $asset)
}
$packages = @{}
foreach ($entry in $index.packages) {
    $version = if ($index.schema_version -eq 1) { $index.package_version } else { $entry.package_version }
    $source = if ($index.schema_version -eq 1) { $index.source_revision } else { $entry.source_revision }
    if ($entry.package_id -cnotmatch '^[a-z0-9][a-z0-9.-]*$' -or $version -cnotmatch '^\d+\.\d+\.\d+$' -or
        $source -cnotmatch '^[a-f0-9]{40}$' -or $entry.sha256 -cnotmatch '^[a-f0-9]{64}$') {
        throw 'Invalid package identity in the existing Windows release index'
    }
    $expectedName = [regex]::Escape("$($entry.package_id)_v$($version)_win_x86_64")
    if ($entry.path -cnotmatch "^$expectedName(?:-[a-f0-9]{12})?\.zip$") {
        throw "Invalid release asset name '$($entry.path)'"
    }
    $asset = $assets[$entry.path]
    if (!$asset -or $asset.size -ne $entry.size_bytes -or ($asset.digest -and $asset.digest -cne "sha256:$($entry.sha256)")) {
        throw "Release asset '$($entry.path)' does not match the existing index"
    }
    $packages.Add($entry.package_id, [ordered]@{
        package_id = $entry.package_id
        package_version = $version
        source_revision = $source
        path = $entry.path
        size_bytes = $entry.size_bytes
        sha256 = $entry.sha256
    })
}

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$upload = @()
$replaced = @()
$updated = $false
$knownZips = @{}
$zips = @(Get-ChildItem -LiteralPath $InputRoot -File -Filter '*.zip' -Recurse)
foreach ($worker in ($Matrix | ConvertFrom-Json).include) {
    $name = "$($worker.id)_v$($worker.version)_win_x86_64.zip"
    $matching = @($zips | Where-Object Name -CEQ $name)
    if ($matching.Count -eq 0) {
        Write-Warning "No successful build artifact for $($worker.id)"
        continue
    }
    if ($matching.Count -ne 1 -or $knownZips.ContainsKey($name)) {
        throw "Expected exactly one artifact for '$name'"
    }
    $knownZips.Add($name, $true)
    $zip = $matching[0]
    $archive = [IO.Compression.ZipFile]::OpenRead($zip.FullName)
    try {
        $manifests = @($archive.Entries | Where-Object FullName -CIn @('worker-package.json', 'worker-manifest.json'))
        if ($manifests.Count -ne 1) { throw "Expected one worker manifest in $name" }
        $reader = [IO.StreamReader]::new($manifests[0].Open())
        try { $manifest = $reader.ReadToEnd() | ConvertFrom-Json } finally { $reader.Dispose() }
        if ($manifest.package_id -cne $worker.id -or $manifest.package_version -cne $worker.version -or
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

    $previous = $packages[$worker.id]
    if ($previous -and $Revision -cne $previous.source_revision) {
        $PSNativeCommandUseErrorActionPreference = $false
        git -C $root merge-base --is-ancestor $Revision $previous.source_revision
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Skipping $($worker.id) because a newer source revision is already published"
            continue
        }
        if ($LASTEXITCODE -ne 1) { throw "Cannot compare the published source revision for $($worker.id)" }
    }
    $digest = (Get-FileHash -LiteralPath $zip.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($previous -and $previous.sha256 -ceq $digest) { continue }

    if ($assets[$name] -and $assets[$name].digest -cne "sha256:$digest") {
        $name = "$([IO.Path]::GetFileNameWithoutExtension($name))-$($digest.Substring(0, 12)).zip"
    }
    if ($assets[$name]) {
        if ($assets[$name].digest -cne "sha256:$digest" -or $assets[$name].size -ne $zip.Length) {
            throw "Asset name collision at '$name'"
        }
    } else {
        Copy-Item -LiteralPath $zip.FullName -Destination (Join-Path $OutputRoot $name)
        $upload += $name
    }
    if ($previous -and $previous.path -cne $name) { $replaced += $previous.path }
    $updated = $true
    $packages[$worker.id] = [ordered]@{
        package_id = $worker.id
        package_version = $worker.version
        source_revision = $Revision
        path = $name
        size_bytes = $zip.Length
        sha256 = $digest
    }
}
if ($zips.Count -ne $knownZips.Count) { throw 'Unexpected Windows package in the build artifacts' }
if (!$updated) { return }

[ordered]@{
    schema_version = 2
    os = 'windows'
    arch = 'x86_64'
    packages = @($packages.Values | Sort-Object { $_.package_id })
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputRoot 'staged-release-index.json') -Encoding utf8NoBOM
[ordered]@{
    upload = @($upload)
    replaced = @($replaced)
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutputRoot 'publication.json') -Encoding utf8NoBOM
