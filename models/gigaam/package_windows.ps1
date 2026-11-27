param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$env:CUDARC_CUDA_VERSION = '12090'
$root = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new staging directory for this package' }
$assets = Join-Path $OutputRoot 'assets'
$package = Join-Path $OutputRoot 'package'
New-Item -ItemType Directory -Path $assets, $package | Out-Null

$archive = Join-Path $OutputRoot 'zerocopy-0.8.63.crate'
curl.exe --fail --location --silent --show-error --max-time 90 'https://static.crates.io/crates/zerocopy/zerocopy-0.8.63.crate' --output $archive
python (Join-Path $root 'tools/restore_vendor_checksums.py') --vendor (Join-Path $root 'vendor') --lockfile (Join-Path $PSScriptRoot 'Cargo.lock') --zerocopy-archive $archive
rustup toolchain install 1.100.0 --profile minimal --no-self-update
$target = Join-Path $OutputRoot 'target'
Push-Location $PSScriptRoot
try {
    cargo '+1.100.0' build --release --locked --offline --no-default-features --features 'flex,cuda,wgpu' --bin gigaam-rs --target-dir $target
} finally {
    Pop-Location
}
$exe = Join-Path $target 'release/gigaam-rs.exe'
$fixtures = Join-Path $PSScriptRoot 'tests/fixtures/backend'
foreach ($kind in @('ctc', 'rnnt')) {
    $checkpoint = Join-Path $OutputRoot "renamed-$kind.safetensors"
    Copy-Item -LiteralPath (Join-Path $fixtures "$kind/model.safetensors") -Destination $checkpoint
    $config = Join-Path $fixtures "$kind/config.json"
    python (Join-Path $PSScriptRoot 'smoke_worker.py') --exe $exe --checkpoint $checkpoint --config $config --model-id "gigaam_v3_$kind" --backend cpu
}

Copy-Item -LiteralPath $exe -Destination $package
foreach ($file in @('LICENSE', 'THIRD_PARTY_NOTICES.md', 'IPC_STDIO_WORKER.md')) {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $file) -Destination $package
}
$models = foreach ($name in @('v3_ctc', 'v3_rnnt', 'v3_e2e_ctc', 'v3_e2e_rnnt', 'v4_ctc', 'v4_rnnt', 'multilingual_ctc', 'multilingual_large_ctc', 'multilingual_v2_rnnt')) {
    $files = @(@{ role = 'checkpoint'; argument = '--checkpoint'; name = "$name.ckpt" })
    if ($name.StartsWith('v3_e2e_')) {
        $files += @{ role = 'tokenizer'; argument = '--tokenizer'; name = "${name}_tokenizer.model" }
    }
    if ($name -in @('v4_rnnt', 'multilingual_v2_rnnt')) {
        $files += @{ role = 'raw_tokenizer'; argument = '--raw-tokenizer'; name = "${name}_raw_tokenizer.model" }
        $files += @{ role = 'e2e_tokenizer'; argument = '--e2e-tokenizer'; name = "${name}_e2e_tokenizer.model" }
    }
    @{
        model_id = "gigaam_$name"
        arguments = @('--model-id', "gigaam_$name")
        artifact_arguments = @($files.argument)
        external_files = $files
    }
}
[ordered]@{
    schema_version = 1
    package_id = 'gigaam-v3'
    package_version = $Version
    source_revision = git -C $root rev-parse HEAD
    os = 'windows'
    arch = 'x86_64'
    worker_protocol = 1
    gigaam_contract_version = 1
    execution_variants = @('burn_flex_cpu_fp32', 'burn_cubecl_cuda_fp32', 'burn_cubecl_wgpu_fp32')
    executable = @{
        path = 'gigaam-rs.exe'
        size_bytes = (Get-Item -LiteralPath $exe).Length
        sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    models = @($models)
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $package 'worker-package.json') -Encoding utf8NoBOM
Compress-Archive -Path (Join-Path $package '*') -DestinationPath (Join-Path $assets "gigaam-v3_v${Version}_win_x86_64.zip") -CompressionLevel Optimal
