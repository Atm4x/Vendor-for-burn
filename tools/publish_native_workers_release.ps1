param(
    [Parameter(Mandatory)][string]$Repository,
    [Parameter(Mandatory)][string]$Tag,
    [Parameter(Mandatory)][string]$Staged,
    [Parameter(Mandatory)][string]$ExistingIndex
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$publication = Get-Content -LiteralPath (Join-Path $Staged 'publication.json') -Raw | ConvertFrom-Json
$indexPath = Join-Path $Staged 'staged-release-index.json'
$index = Get-Content -LiteralPath $indexPath -Raw | ConvertFrom-Json
foreach ($name in $publication.upload) {
    $entry = @($index.packages | Where-Object path -CEQ $name)
    $path = Join-Path $Staged $name
    if ($entry.Count -ne 1 -or (Get-Item -LiteralPath $path).Length -ne $entry[0].size_bytes -or
        (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $entry[0].sha256) {
        throw "Staged asset '$name' does not match the index"
    }
    gh release upload $Tag $path --repo $Repository
}

$indexDigest = (Get-FileHash -LiteralPath $indexPath -Algorithm SHA256).Hash.ToLowerInvariant()
$PSNativeCommandUseErrorActionPreference = $false
for ($attempt = 1; $attempt -le 3; $attempt++) {
    gh release upload $Tag $indexPath --repo $Repository --clobber
    $digest = gh api "repos/$Repository/releases/tags/$Tag" --jq '.assets[] | select(.name == "staged-release-index.json") | .digest'
    if ($digest -ceq "sha256:$indexDigest") { break }
    if ($attempt -eq 3) {
        if (!$digest) { gh release upload $Tag $ExistingIndex --repo $Repository }
        throw 'Could not publish and verify the new Windows release index'
    }
    Start-Sleep -Seconds (2 * $attempt)
}

$PSNativeCommandUseErrorActionPreference = $true
gh release edit $Tag --repo $Repository --title 'Native workers (Windows x64)' --notes 'Windows packages are updated independently. The signed staged-release-index.json records the active version, source revision and SHA-256 for each ZIP. Model weights are installed separately.'

$PSNativeCommandUseErrorActionPreference = $false
foreach ($name in $publication.replaced) {
    gh release delete-asset $Tag $name --repo $Repository --yes
    if ($LASTEXITCODE -ne 0) { Write-Warning "Could not remove the superseded asset '$name'" }
}
