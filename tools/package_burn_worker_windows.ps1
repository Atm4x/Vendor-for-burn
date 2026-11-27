param(
    [Parameter(Mandatory)][ValidateSet('omnivoice', 'whisper', 'semantic')][string]$Family,
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][string]$OutputRoot,
    [string]$TargetDir
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new staging directory' }
$root = Split-Path $PSScriptRoot -Parent
$project = Join-Path $root "models/$Family"
$binary = "$Family-worker"
$OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)
$package = Join-Path $OutputRoot 'package'
$assets = Join-Path $OutputRoot 'assets'
New-Item -ItemType Directory -Path $package, $assets | Out-Null
if (!$TargetDir) { $TargetDir = Join-Path $OutputRoot 'target' }
$TargetDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($TargetDir)
if ($env:CI -eq 'true') {
    $archive = Join-Path $OutputRoot 'zerocopy-0.8.63.crate'
    curl.exe --fail --location --silent --show-error --max-time 90 'https://static.crates.io/crates/zerocopy/zerocopy-0.8.63.crate' --output $archive
    python (Join-Path $root 'tools/restore_vendor_checksums.py') --vendor (Join-Path $root 'vendor') --lockfile (Join-Path $project 'Cargo.lock') --zerocopy-archive $archive
    rustup toolchain install 1.100.0 --profile minimal --no-self-update
}

$env:CUDARC_CUDA_VERSION = '12090'
Push-Location $project
try {
    cargo '+1.100.0' build --release --locked --offline --bin $binary --target-dir $TargetDir
} finally { Pop-Location }

$exe = Join-Path $TargetDir "release/$binary.exe"
Copy-Item -LiteralPath $exe -Destination $package
$files = @('LICENSE', 'THIRD_PARTY_NOTICES.md', 'IPC_STDIO_WORKER.md', 'install-plan.json')
$variants = @('burn_flex_cpu_fp32', 'burn_cubecl_cuda_fp32', 'burn_cubecl_wgpu_fp32')
if ($Family -eq 'semantic') {
    $files += @('CANDLE-LICENSE-MIT', 'CANDLE-LICENSE-APACHE', 'create_install_plan.py')
    $variants += 'candle_cpu_fp32'
}
foreach ($name in $files) {
    Copy-Item -LiteralPath (Join-Path $project $name) -Destination $package
}

$plan = Get-Content -LiteralPath (Join-Path $project 'install-plan.json') -Raw | ConvertFrom-Json
$models = foreach ($model in $plan.models) {
    [ordered]@{
        model_id = $model.model_id
        service = $model.service
        launch_arguments = @($model.launch_arguments)
        artifact_arguments = @($model.downloads.argument)
    }
}
[ordered]@{
    schema_version = 1
    package_id = $Family
    package_version = $Version
    source_revision = git -C $root rev-parse HEAD
    os = 'windows'
    arch = 'x86_64'
    worker_protocol = 1
    execution_variants = $variants
    executable = @{
        path = "$binary.exe"
        size_bytes = (Get-Item -LiteralPath $exe).Length
        sha256 = (Get-FileHash -LiteralPath $exe).Hash.ToLowerInvariant()
    }
    models = @($models)
    install_plan = 'install-plan.json'
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $package 'worker-package.json') -Encoding utf8NoBOM

$zip = Join-Path $assets "$($Family)_v$($Version)_win_x86_64.zip"
Compress-Archive -Path (Join-Path $package '*') -DestinationPath $zip -CompressionLevel Optimal
