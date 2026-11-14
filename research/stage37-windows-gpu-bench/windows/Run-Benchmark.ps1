param(
    [ValidateSet('all','cuda','wgpu')][string]$Backend = 'all',
    [int]$Warmup = 3,
    [int]$Iterations = 10,
    [switch]$InstallTorchCu128,
    [switch]$SkipTorch
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$reportDir = Join-Path $PSScriptRoot 'reports'
New-Item -ItemType Directory -Force -Path $reportDir | Out-Null

function Save-CommandOutput([string]$Command, [string[]]$Arguments, [string]$Destination) {
    $commandInfo = Get-Command $Command -ErrorAction SilentlyContinue
    if ($null -eq $commandInfo) {
        "NOT FOUND: $Command" | Set-Content -Path $Destination -Encoding UTF8
        Write-Host "$Command : not found"
        return $false
    }
    & $commandInfo.Source @Arguments 2>&1 | Out-File -FilePath $Destination -Encoding utf8
    Write-Host "$Command : exit code $LASTEXITCODE"
    return $LASTEXITCODE -eq 0
}

function Invoke-LoggedNative([string]$Executable, [string]$Arguments, [string]$LogPrefix) {
    $stdout = "$LogPrefix.stdout.log"
    $stderr = "$LogPrefix.stderr.log"
    $process = Start-Process -FilePath $Executable -ArgumentList $Arguments -WorkingDirectory $PSScriptRoot -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if (Test-Path $stdout) { Get-Content $stdout | ForEach-Object { Write-Host $_ } }
    if (Test-Path $stderr) { Get-Content $stderr | ForEach-Object { Write-Host $_ } }
    return $process.ExitCode
}

Write-Host 'Collecting device and runtime diagnostics...'
$null = Save-CommandOutput -Command 'nvidia-smi' -Arguments @('-q') -Destination (Join-Path $reportDir 'nvidia-smi.txt')
$null = Save-CommandOutput -Command 'nvcc' -Arguments @('--version') -Destination (Join-Path $reportDir 'nvcc.txt')
$envInfo = @{
    utc = (Get-Date).ToUniversalTime().ToString('o')
    os = [System.Environment]::OSVersion.VersionString
    architecture = $env:PROCESSOR_ARCHITECTURE
    CUDA_PATH = $env:CUDA_PATH
    PATH_CUDA = @($env:PATH.Split(';') | Where-Object { $_ -match 'CUDA' })
    comparison_method = 'kernel dispatch plus synchronous CPU readback (not pure device kernel latency)'
}
$envInfo | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $reportDir 'environment.json') -Encoding UTF8

$backends = if ($Backend -eq 'all') { @('cuda','wgpu') } else { @($Backend) }
foreach ($name in $backends) {
    $exePath = Join-Path $PSScriptRoot ("rvc-cubecl-$name.exe")
    if (-not (Test-Path $exePath)) {
        Write-Warning "Backend binary unavailable: $exePath. Check the build-status.txt file."
        continue
    }
    Write-Host "Running CubeCL $name ..."
    $report = Join-Path $reportDir "$name.json"
    $tensorDir = Join-Path $reportDir "cubecl-$name-tensors"
    $benchmarkArgs = "--backend $name --iterations $Iterations --warmup $Warmup --json `"$report`" --parity-dir `"$tensorDir`""
    $exitCode = Invoke-LoggedNative -Executable $exePath -Arguments $benchmarkArgs -LogPrefix (Join-Path $reportDir $name)
    if ($exitCode -ne 0) {
        Write-Warning "CubeCL $name failed. Raw log preserved; no speedup claimed."
    }
}

if (-not $SkipTorch) {
    $python = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($null -eq $python) {
        $python = Get-Command 'python' -ErrorAction SilentlyContinue
        $pyArgs = @()
    } else {
        $pyArgs = @('-3')
    }
    if ($python) {
        if ($InstallTorchCu128) {
            Write-Host 'Installing PyTorch CUDA 12.8 wheels (requested explicitly)...'
            & $python.Source @pyArgs -m pip install 'torch==2.11.0' --index-url 'https://download.pytorch.org/whl/cu128'
            if ($LASTEXITCODE -ne 0) { Write-Warning 'PyTorch installation failed; Python benchmark may be unavailable.' }
        }
        & $python.Source @pyArgs -c 'import torch; print(torch.__version__)' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $torchReport = Join-Path $reportDir 'pytorch.json'
            $torchTensorDir = Join-Path $reportDir 'pytorch-tensors'
            $pythonArgs = ((@($pyArgs) + @('.\torch_cudnn_bench.py', '--warmup', "$Warmup", '--iterations', "$Iterations", '--report', "`"$torchReport`"", '--tf32', 'disable', '--parity-dir', "`"$torchTensorDir`"")) -join ' ')
            $exitCode = Invoke-LoggedNative -Executable $python.Source -Arguments $pythonArgs -LogPrefix (Join-Path $reportDir 'pytorch')
            if ($exitCode -ne 0) { Write-Warning 'PyTorch benchmark failed; see reports/pytorch.stderr.log.' }
        } else {
            Write-Warning 'PyTorch is not installed. Use -InstallTorchCu128 explicitly to install a CUDA-enabled wheel.'
        }
        foreach ($name in $backends) {
            $cubeReport = Join-Path $reportDir "$name.json"
            $torchReport = Join-Path $reportDir 'pytorch.json'
            if ((Test-Path $cubeReport) -and (Test-Path $torchReport)) {
                & $python.Source @pyArgs .\merge_reports.py --cubecl $cubeReport --torch $torchReport --out (Join-Path $reportDir "$name-vs-pytorch.json")
                if ($LASTEXITCODE -ne 0) { Write-Warning "Probe parity check failed for $name; inspect output JSON." }
                $cubeTensorDir = Join-Path $reportDir "cubecl-$name-tensors"
                if ((Test-Path $cubeTensorDir) -and (Test-Path (Join-Path $reportDir 'pytorch-tensors'))) {
                    & $python.Source @pyArgs .\compare_tensor_dumps.py --cubecl-dir $cubeTensorDir --torch-dir (Join-Path $reportDir 'pytorch-tensors') --out (Join-Path $reportDir "$name-full-tensor-parity.json")
                    if ($LASTEXITCODE -ne 0) { Write-Warning "Full tensor parity failed for $name; inspect the report." }
                }
            }
        }
    } else {
        Write-Warning 'Python not found. PyTorch comparison skipped; CubeCL tests are independent.'
    }
}
$archive = Join-Path $PSScriptRoot 'RVC_GPU_BENCH_RESULTS.zip'
if (Test-Path $archive) { Remove-Item $archive -Force }
Compress-Archive -Path (Join-Path $reportDir '*') -DestinationPath $archive -Force
Write-Host "Completed. Send back: $archive"
