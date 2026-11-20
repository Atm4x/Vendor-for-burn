param(
    [string]$Python = "python",
    [int]$Iterations = 16,
    [int]$Warmup = 5,
    [string]$Frames = "64,256,1024"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "profile_nsf_graph_stage52.exe"
$checkpoint = Join-Path $PSScriptRoot "synthetic_nsf_decoder.pth"
$fixture = Join-Path $PSScriptRoot "nsf_decoder.json"
$torch = Join-Path $PSScriptRoot "torch_nsf_stage53.py"
$compare = Join-Path $PSScriptRoot "compare_nsf_stage53.py"
$reports = Join-Path $PSScriptRoot "stage53_results"
$archive = Join-Path $PSScriptRoot "NSF_TORCH_CUDNN_STAGE53_RESULTS.zip"
if ($Iterations -lt 4 -or $Warmup -lt 0) { throw "Use >=4 iterations and >=0 warmup" }
foreach ($file in @($exe, $checkpoint, $fixture, $torch, $compare)) {
    if (-not (Test-Path -LiteralPath $file)) { throw "Missing benchmark component: $file" }
}
New-Item -ItemType Directory -Force -Path $reports | Out-Null
& $Python -c "import torch; assert torch.cuda.is_available(), 'PyTorch CUDA required'; print('PyTorch',torch.__version__, 'cuDNN',torch.backends.cudnn.version(), 'GPU',torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw "Python with CUDA-enabled PyTorch is required; run in the existing TTS environment or specify -Python" }
try { & nvidia-smi -q | Set-Content (Join-Path $reports "nvidia-smi.txt") } catch { $_ | Out-String | Set-Content (Join-Path $reports "nvidia-smi-error.txt") }
$oldPrepack = [Environment]::GetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", "Process")
$oldKernel = [Environment]::GetEnvironmentVariable("RVC_NSF_CONV_KERNEL", "Process")
try {
    foreach ($round in @(1, 2)) {
        $order = if ($round -eq 1) { @("torch", "rust_prepacked", "rust_baseline") } else { @("rust_baseline", "rust_prepacked", "torch") }
        foreach ($name in $order) {
            $id = "round${round}_${name}"
            $stdout = Join-Path $reports "$id.stdout.log"
            $stderr = Join-Path $reports "$id.stderr.log"
            $json = Join-Path $reports "$id.json"
            $prefix = "stage53_results/$id"
            Write-Host "Starting $id"
            if ($name -eq "torch") {
                $arguments = @(
                    $torch, "--checkpoint", $checkpoint, "--fixture", $fixture,
                    "--frames", $Frames, "--iterations", "$Iterations", "--warmup", "$Warmup",
                    "--report", $json, "--dump-prefix", (Join-Path $reports $id)
                )
                $priorActionPreference = $ErrorActionPreference
                try {
                    $ErrorActionPreference = "Continue"
                    & $Python @arguments 1> $stdout 2> $stderr
                    $exitCode = $LASTEXITCODE
                }
                finally { $ErrorActionPreference = $priorActionPreference }
                if ($exitCode -ne 0) { throw "$id failed (Python/cuDNN); see $stderr" }
                Get-Content -LiteralPath $stdout
            } else {
                Remove-Item Env:RVC_NSF_CONV_KERNEL -ErrorAction SilentlyContinue
                if ($name -eq "rust_prepacked") { $env:RVC_CONV_WEIGHT_PREPACK = "1" }
                else { Remove-Item Env:RVC_CONV_WEIGHT_PREPACK -ErrorAction SilentlyContinue }
                $arguments = @(
                    "--backend", "cuda", "--checkpoint", "synthetic_nsf_decoder.pth",
                    "--execution", "sequential", "--frames", $Frames,
                    "--iterations", "$Iterations", "--dump-prefix", $prefix
                )
                $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $json -RedirectStandardError $stderr -PassThru -Wait -NoNewWindow
                if ($process.ExitCode -ne 0) { throw "$id failed (Rust/CubeCL); see $stderr" }
            }
        }
    }
    & $Python $compare --results $reports --out (Join-Path $reports "summary.json")
    if ($LASTEXITCODE -ne 0) { throw "Stage53 full-tensor verification failed" }
}
finally {
    [Environment]::SetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", $oldPrepack, "Process")
    [Environment]::SetEnvironmentVariable("RVC_NSF_CONV_KERNEL", $oldKernel, "Process")
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    Compress-Archive -Path (Join-Path $reports "*") -DestinationPath $archive -CompressionLevel Optimal
    Write-Host "Results: $archive"
}
