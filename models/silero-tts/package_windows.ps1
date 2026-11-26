param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][string]$OutputRoot
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new staging directory' }
$assets = Join-Path $OutputRoot 'assets'
$package = Join-Path $OutputRoot 'package'
New-Item -ItemType Directory -Path $assets, $package | Out-Null
$archive = Join-Path $OutputRoot 'zerocopy-0.8.63.crate'
curl.exe --fail --location --silent --show-error --max-time 90 'https://static.crates.io/crates/zerocopy/zerocopy-0.8.63.crate' --output $archive
python (Join-Path $root 'tools/restore_vendor_checksums.py') --vendor (Join-Path $root 'vendor') --lockfile (Join-Path $PSScriptRoot 'Cargo.lock') --zerocopy-archive $archive
rustup toolchain install 1.100.0 --profile minimal --no-self-update
$manifest = Join-Path $PSScriptRoot 'Cargo.toml'
$target = Join-Path $OutputRoot 'target'
Push-Location $PSScriptRoot
try {
    cargo '+1.100.0' build --offline --release --locked --manifest-path $manifest --no-default-features --features reference-flex --bin silero-tts --target-dir $target
} finally { Pop-Location }
$exe = Join-Path $target 'release/silero-tts.exe'
Copy-Item -LiteralPath $exe -Destination $package
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'IPC_STDIO_WORKER.md') -Destination $package
[ordered]@{
    schema_version = 1
    package_id = 'silero-tts'
    package_version = $Version
    source_revision = git -C $PSScriptRoot rev-parse HEAD
    os = 'windows'
    arch = 'x86_64'
    worker_protocol = 1
    silero_contract_version = 1
    backends = @('flex')
    execution_variants = @('burn_flex_cpu_fp32')
    executable = @{
        path = 'silero-tts.exe'
        size_bytes = (Get-Item -LiteralPath $exe).Length
        sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    models = @(
        @{ model_id = 'v3_en'; artifact_arguments = @('--checkpoint'); checkpoint = @{
            url = 'https://models.silero.ai/models/tts/en/v3_en.pt'; size_bytes = 57194546
            sha256 = '02b71034d9f13bc4001195017bac9db1c6bb6115e03fea52983e8abcff13b665'
        } },
        @{ model_id = 'v5_ru'; artifact_arguments = @('--checkpoint'); checkpoint = @{
            url = 'https://models.silero.ai/models/tts/ru/v5_ru.pt'; size_bytes = 145382211
            sha256 = '7ba04d42340fe0398042eed2e0d12d62e23096d626b1b9feff4dcb1309197ab4'
        } }
    )
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $package 'worker-package.json') -Encoding utf8NoBOM
$zip = Join-Path $assets "silero-tts_v${Version}_win_x86_64.zip"
Compress-Archive -Path (Join-Path $package '*') -DestinationPath $zip -CompressionLevel Optimal
$installed = Join-Path $OutputRoot 'installed'
Expand-Archive -LiteralPath $zip -DestinationPath $installed
$manifest = Get-Content -LiteralPath (Join-Path $installed 'worker-package.json') -Raw | ConvertFrom-Json
$installedExe = Join-Path $installed $manifest.executable.path
if ((Get-FileHash -LiteralPath $installedExe -Algorithm SHA256).Hash.ToLowerInvariant() -cne $manifest.executable.sha256) { throw 'Invalid executable hash' }
$weights = @{}
foreach ($model in $manifest.models) {
    $path = Join-Path $OutputRoot ($model.model_id + '-renamed.pt')
    curl.exe --fail --location --retry 3 $model.checkpoint.url --output $path
    if ((Get-Item -LiteralPath $path).Length -ne $model.checkpoint.size_bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $model.checkpoint.sha256) { throw 'Invalid checkpoint' }
    $weights[$model.model_id] = $path
}
$env:RAYON_NUM_THREADS = '4'
python (Join-Path $PSScriptRoot 'tools/smoke_native_worker.py') $installedExe --en $weights['v3_en'] --ru $weights['v5_ru']
