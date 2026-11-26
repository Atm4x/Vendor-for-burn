param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$env:CUDARC_CUDA_VERSION = '12090'
$llvm = Join-Path $env:ProgramFiles 'LLVM/bin'
if (Test-Path $llvm) { $env:LIBCLANG_PATH = $llvm }
$env:BINDGEN_EXTRA_CLANG_ARGS = '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new staging directory for this package' }
$assets = Join-Path $OutputRoot 'assets'
$package = Join-Path $OutputRoot 'package'
New-Item -ItemType Directory -Path $assets, $package | Out-Null
rustup toolchain install 1.100.0 --profile minimal --no-self-update
$manifest = Join-Path $PSScriptRoot 'Cargo.toml'
$target = Join-Path $OutputRoot 'target'
Push-Location $OutputRoot
try {
    cargo '+1.100.0' build --release --locked --manifest-path $manifest --no-default-features --features 'flex,cuda,wgpu,extended-audio,persistent-gru,dynamic-conv' --bin dettato-worker --target-dir $target
} finally {
    Pop-Location
}
$exe = Join-Path $target 'release/dettato-worker.exe'
python (Join-Path $PSScriptRoot 'smoke_worker.py') $exe
Copy-Item -LiteralPath $exe -Destination $package
foreach ($name in 'LICENSE', 'NOTICE', 'IPC_STDIO_WORKER.md') {
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot $name) -Destination $package
}
$models = foreach ($name in @('transcriber', 'evidence', 'grouper', 'drums')) {
    @{ role = "$name-config"; argument = "--$name-config" }
    @{ role = "$name-weights"; argument = "--$name-weights" }
}
[ordered]@{
    schema_version = 1
    package_id = 'dettato'
    package_version = $Version
    source_revision = git -C $PSScriptRoot rev-parse HEAD
    os = 'windows'
    arch = 'x86_64'
    worker_protocol = 1
    dettato_contract_version = 1
    backends = @('cpu', 'cuda', 'wgpu')
    executable = @{
        path = 'dettato-worker.exe'
        size_bytes = (Get-Item -LiteralPath $exe).Length
        sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    external_files = @($models)
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $package 'worker-package.json') -Encoding utf8NoBOM
Compress-Archive -Path (Join-Path $package '*') -DestinationPath (Join-Path $assets "dettato_v${Version}_win_x86_64.zip") -CompressionLevel Optimal
