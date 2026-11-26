param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$env:CUDARC_CUDA_VERSION = '12090'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new staging directory for this package' }
New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$archive = Join-Path $OutputRoot 'zerocopy-0.8.63.crate'
curl.exe --fail --location --silent --show-error --max-time 90 'https://static.crates.io/crates/zerocopy/zerocopy-0.8.63.crate' --output $archive
rustup toolchain install 1.100.0 --profile minimal --no-self-update
$target = Join-Path $OutputRoot 'target'
Push-Location $root
try {
    python tools/restore_vendor_checksums.py --zerocopy-archive $archive
    cargo '+1.100.0' build --release --locked --offline -p tts-with-rvc-rs --no-default-features --features 'cpu,cuda,wgpu' --bin rvc-worker --target-dir $target
} finally {
    Pop-Location
}
$exe = Join-Path $target 'release/rvc-worker.exe'
python (Join-Path $PSScriptRoot 'smoke_worker.py') --exe $exe --backend cpu
$staging = Join-Path $OutputRoot 'staged'
& (Join-Path $PSScriptRoot 'package_windows.ps1') -Version $Version -Backend @('cpu', 'cuda', 'wgpu') -ExePath $exe -OutputRoot $staging
$assets = Join-Path $OutputRoot 'assets'
New-Item -ItemType Directory -Path $assets | Out-Null
Compress-Archive -Path (Join-Path $staging 'package/*') -DestinationPath (Join-Path $assets "rvc-v2_v${Version}_win_x86_64.zip") -CompressionLevel Optimal
