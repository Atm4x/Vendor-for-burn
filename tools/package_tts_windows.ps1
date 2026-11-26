param(
    [Parameter(Mandatory)][ValidatePattern('^\d+\.\d+\.\d+$')][string]$Version,
    [Parameter(Mandatory)][string]$OutputRoot,
    [Parameter(Mandatory)][ValidateSet('fish-speech-1.5', 'f5-tts')][string]$Family
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true
$root = Split-Path $PSScriptRoot -Parent
$definition = switch ($Family) {
    'fish-speech-1.5' {
        @{
            project = 'models/fish-speech'
            binary = 'fish-speech-worker'
            models = @(@{
                model_id = 'fish_speech_1_5'
                artifact_arguments = @('--semantic-dir', '--firefly-dir', '--firefly-config')
            })
        }
    }
    'f5-tts' {
        @{
            project = 'models/f5_tts'
            binary = 'f5-tts-worker'
            models = @(
                @{ model_id = 'f5_tts_ru'; launch_arguments = @('--model-id', 'f5_tts_ru'); artifact_arguments = @('--dit-weights', '--vocos-weights', '--vocab') }
                @{ model_id = 'f5_tts_cross_lingual'; launch_arguments = @('--model-id', 'f5_tts_cross_lingual'); artifact_arguments = @('--dit-weights', '--vocos-weights', '--vocab', '--speed-weights') }
            )
        }
    }
}
if (Test-Path -LiteralPath $OutputRoot) { throw 'Use a new staging directory for this package' }
$assets = Join-Path $OutputRoot 'assets'
$package = Join-Path $OutputRoot 'package'
New-Item -ItemType Directory -Path $assets, $package | Out-Null
$llvm = Join-Path $env:ProgramFiles 'LLVM/bin'
if (Test-Path $llvm) { $env:LIBCLANG_PATH = $llvm }
$env:BINDGEN_EXTRA_CLANG_ARGS = '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'
$env:CUDARC_CUDA_VERSION = '12090'
rustup toolchain install 1.100.0 --profile minimal --no-self-update
$project = Join-Path $root $definition.project
$manifest = Join-Path $project 'Cargo.toml'
$target = Join-Path $OutputRoot 'target'
Push-Location $OutputRoot
try {
    cargo '+1.100.0' build --release --locked --manifest-path $manifest --features worker --bin $definition.binary --target-dir $target
} finally {
    Pop-Location
}
$exe = Join-Path $target "release/$($definition.binary).exe"
python (Join-Path $PSScriptRoot 'smoke_tts_worker.py') $exe $Family
Copy-Item -LiteralPath $exe -Destination $package
[ordered]@{
    schema_version = 1
    package_id = $Family
    package_version = $Version
    source_revision = git -C $root rev-parse HEAD
    os = 'windows'
    arch = 'x86_64'
    worker_protocol = 1
    tts_contract_version = 1
    backends = @('cuda', 'wgpu')
    executable = @{
        path = "$($definition.binary).exe"
        size_bytes = (Get-Item -LiteralPath $exe).Length
        sha256 = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    models = $definition.models
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $package 'worker-package.json') -Encoding utf8NoBOM
Compress-Archive -Path (Join-Path $package '*') -DestinationPath (Join-Path $assets "${Family}_v${Version}_win_x86_64.zip") -CompressionLevel Optimal
