param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][string]$OutputRoot,
    [switch]$Compact
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$modelRoot = $PSScriptRoot
$repositoryRoot = (Resolve-Path (Join-Path $modelRoot '../..')).Path
$definition = Get-Content -Raw (Join-Path $modelRoot 'distribution.json') | ConvertFrom-Json
$versionMatch = Select-String -Path (Join-Path $modelRoot 'Cargo.toml') -Pattern '^version = "([0-9.]+)"$' | Select-Object -First 1
if (!$Compact -and $versionMatch.Matches[0].Groups[1].Value -ne $Version) {
    throw 'Release version must match Cargo.toml'
}

if (Test-Path -LiteralPath $OutputRoot) {
    throw 'Use a new staging directory for this package'
}
$assets = Join-Path $OutputRoot 'assets'
$package = Join-Path $OutputRoot 'package'
$modelFiles = Join-Path $package 'model'
$runtimeFiles = Join-Path $package 'runtime'
$runtimeFile = Join-Path $runtimeFiles $definition.soxr.path
New-Item -ItemType Directory -Path $assets, $modelFiles, (Split-Path $runtimeFile -Parent) | Out-Null

function Test-FileDigest($Path, $Sha256) {
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {
        throw "SHA-256 mismatch for $Path"
    }
}

function ConvertTo-FileReference($Path, $Relative) {
    return [ordered]@{
        path = $Relative
        size_bytes = (Get-Item -LiteralPath $Path).Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

$soxrArchive = Join-Path $OutputRoot 'soxr-source.zip'
curl.exe --fail --location --silent --show-error --max-time 180 $definition.soxr.source_url --output $soxrArchive
Test-FileDigest $soxrArchive $definition.soxr.source_sha256
Expand-Archive -LiteralPath $soxrArchive -DestinationPath $OutputRoot
$soxrSource = Join-Path $OutputRoot "soxr-$($definition.soxr.source_revision)"
$soxrBuild = Join-Path $OutputRoot 'soxr-build'
cmake -S $soxrSource -B $soxrBuild -G 'Visual Studio 17 2022' -A x64 '-DBUILD_SHARED_LIBS=ON' '-DWITH_OPENMP=OFF' '-DWITH_LSR_BINDINGS=OFF' '-DBUILD_TESTS=OFF' '-DBUILD_EXAMPLES=OFF' '-DCMAKE_POLICY_VERSION_MINIMUM=3.5'
cmake --build $soxrBuild --config Release --parallel 2
Copy-Item -LiteralPath (Join-Path $soxrBuild 'bin/Release/soxr.dll') -Destination $runtimeFile
Copy-Item -LiteralPath (Join-Path $soxrSource 'COPYING.LGPL') -Destination (Join-Path $package 'COPYING.SoXR.LGPL')
Copy-Item -LiteralPath $soxrArchive -Destination (Join-Path $package 'soxr-source.zip')
Copy-Item -LiteralPath (Join-Path $modelRoot 'LICENSE') -Destination (Join-Path $package 'LICENSE.Beat-This')

$vendorArchive = Join-Path $OutputRoot 'zerocopy-0.8.63.crate'
curl.exe --fail --location --silent --show-error --max-time 90 'https://static.crates.io/crates/zerocopy/zerocopy-0.8.63.crate' --output $vendorArchive
python (Join-Path $repositoryRoot 'tools/restore_vendor_checksums.py') --vendor (Join-Path $repositoryRoot 'vendor') --lockfile (Join-Path $modelRoot 'Cargo.lock') --zerocopy-archive $vendorArchive
rustup toolchain install 1.100.0 --profile minimal --no-self-update
$nativeHost = rustup run 1.100.0 rustc -vV | Select-String '^host: x86_64-pc-windows-msvc$'
if (!$nativeHost) { throw 'This package requires the native Windows x64 MSVC toolchain' }
Push-Location $modelRoot
try {
    cargo '+1.100.0' test --release --locked --offline -j 2
    cargo '+1.100.0' build --release --locked --offline -j 2
} finally {
    Pop-Location
}

$exe = Join-Path $modelFiles 'beat-this-rs.exe'
Copy-Item -LiteralPath (Join-Path $modelRoot 'target/release/beat-this-rs.exe') -Destination $exe

if ($Compact) {
    $models = @($definition) + @($definition.additional_models)
    $modelEntries = foreach ($model in $models) {
        $weights = $model.weights
        $checkpoint = Join-Path $OutputRoot ("test-" + $weights.name)
        curl.exe --fail --location --silent --show-error --max-time 180 $weights.url --output $checkpoint
        Test-FileDigest $checkpoint $weights.sha256
        if ((Get-Item -LiteralPath $checkpoint).Length -ne $weights.size_bytes) {
            throw "Checkpoint size does not match for $($model.model_id)"
        }
        python (Join-Path $modelRoot 'smoke_worker.py') --exe $exe --checkpoint $checkpoint --runtime-dir $runtimeFiles --model-id $model.model_id
        $entry = [ordered]@{
            model_id = $model.model_id
            display_name = $model.display_name
            arguments = @('--model-id', $model.model_id)
            artifact_arguments = @('--checkpoint')
            weights = @{
                role = 'checkpoint'
                argument = '--checkpoint'
                file = ConvertTo-FileReference $checkpoint $weights.name
                url = $weights.url
            }
        }
        Remove-Item -LiteralPath $checkpoint
        $entry
    }
    [ordered]@{
        schema_version = 1
        package_id = 'beat-this'
        package_version = $Version
        source_revision = git -C $repositoryRoot rev-parse HEAD
        os = 'windows'
        arch = 'x86_64'
        execution_variant = $definition.execution_variant
        worker_protocol = $definition.worker_protocol
        beats_contract_version = $definition.beats_contract_version
        executable = ConvertTo-FileReference $exe 'model/beat-this-rs.exe'
        runtime_files = @(ConvertTo-FileReference $runtimeFile "runtime/$($definition.soxr.path)")
        models = @($modelEntries)
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $package 'worker-package.json') -Encoding utf8NoBOM
    $zip = Join-Path $assets "beat-this_v${Version}_win_x86_64.zip"
    Compress-Archive -Path (Join-Path $package '*') -DestinationPath $zip -CompressionLevel Optimal
    return
}

$checkpoint = Join-Path $modelFiles $definition.weights.name
curl.exe --fail --location --silent --show-error --max-time 180 $definition.weights.url --output $checkpoint
Test-FileDigest $checkpoint $definition.weights.sha256
if ((Get-Item -LiteralPath $checkpoint).Length -ne $definition.weights.size_bytes) {
    throw 'Checkpoint size does not match'
}
python (Join-Path $modelRoot 'smoke_worker.py') --exe $exe --model-dir $modelFiles --runtime-dir $runtimeFiles

$prefix = "beat-this-small1-v$Version-windows-x64-flex"
$sourceRevision = git -C $repositoryRoot rev-parse HEAD
$manifest = [ordered]@{
    schema_version = 1
    model_id = $definition.model_id
    display_name = $definition.display_name
    package_version = $Version
    os = 'windows'
    arch = 'x86_64'
    protocol_version = $definition.worker_protocol
    backends = @('cpu')
    executable = ConvertTo-FileReference $exe 'beat-this-rs.exe'
    assets = @(ConvertTo-FileReference $checkpoint $definition.weights.name)
    runtime_files = @(@{ backends = @('cpu'); file = ConvertTo-FileReference $runtimeFile $definition.soxr.path })
}
$manifestFile = Join-Path $assets "$prefix.native-model.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestFile -Encoding utf8NoBOM
Copy-Item -LiteralPath $manifestFile -Destination (Join-Path $modelFiles 'native-model.json')
$weights = [ordered]@{
    role = 'checkpoint'
    file = $manifest.assets[0]
    url = $definition.weights.url
}
# Download the checkpoint from its verified upstream source at install time.
Remove-Item -LiteralPath $checkpoint
$publishedExe = Join-Path $assets "$prefix.exe"
Copy-Item -LiteralPath $exe -Destination $publishedExe
$zip = Join-Path $assets "$prefix.zip"
Compress-Archive -Path (Join-Path $package '*') -DestinationPath $zip -CompressionLevel Optimal

[ordered]@{
    schema_version = 1
    model_id = $definition.model_id
    display_name = $definition.display_name
    description = $definition.description
    version = $Version
    variant = $definition.execution_variant
    source_revision = $sourceRevision
    worker_protocol = $definition.worker_protocol
    beats_contract_version = $definition.beats_contract_version
    package = ConvertTo-FileReference $zip "$prefix.zip"
    binary = ConvertTo-FileReference $publishedExe "$prefix.exe"
    manifest = ConvertTo-FileReference $manifestFile "$prefix.native-model.json"
    weights = $weights
    runtime = $manifest.runtime_files
} | ConvertTo-Json -Depth 9 | Set-Content -LiteralPath (Join-Path $assets "$prefix.install-plan.json") -Encoding utf8NoBOM

Get-ChildItem -LiteralPath $assets -File | Sort-Object Name | ForEach-Object {
    "$((Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant())  $($_.Name)"
} | Set-Content -LiteralPath (Join-Path $assets "$prefix.sha256") -Encoding ascii
