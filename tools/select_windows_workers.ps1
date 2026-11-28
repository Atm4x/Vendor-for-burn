param(
    [string]$Before,
    [Parameter(Mandatory)][ValidatePattern('^[a-f0-9]{40}$')][string]$Revision,
    [string]$Worker,
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$root = Split-Path -Parent $PSScriptRoot
$workers = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'windows-worker-packages.json') -Raw | ConvertFrom-Json
if ($Worker) {
    $selected = @($workers | Where-Object id -CEQ $Worker)
    if ($selected.Count -ne 1) { throw "Unknown Windows worker '$Worker'" }
} else {
    if ($Version) { throw 'A version override requires a single worker' }
    if ($Before -notmatch '^[a-f0-9]{40}$' -or $Before -eq ('0' * 40)) {
        throw 'A push must supply its previous main revision'
    }
    $paths = @(git -C $root diff --no-renames --name-only $Before $Revision -- models/)
    $selected = @($workers | Where-Object {
        $prefix = $_.directory + '/'
        $paths.Where({ $_.StartsWith($prefix, [StringComparison]::Ordinal) }, 'First').Count
    })
}

$matrix = @(foreach ($package in $selected) {
    $versionPath = Join-Path $root ($package.directory + '/worker-version.txt')
    $chosen = if ($Version) {
        $Version
    } elseif (Test-Path -LiteralPath $versionPath -PathType Leaf) {
        (Get-Content -LiteralPath $versionPath -Raw).Trim()
    } else {
        '1.0.0'
    }
    if ($chosen -cnotmatch '^\d+\.\d+\.\d+$') {
        throw "Invalid package version '$chosen' for $($package.id)"
    }
    [ordered]@{
        id = $package.id
        script = $package.script
        family = [string]$package.family
        compact = [bool]$package.compact
        version = $chosen
    }
})

ConvertTo-Json -InputObject @{ include = $matrix } -Depth 4 -Compress
