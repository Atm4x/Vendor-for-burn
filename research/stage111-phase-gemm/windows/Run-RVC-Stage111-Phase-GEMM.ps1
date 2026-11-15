param(
    [ValidateSet('none','ncu')][string]$Profiler = 'none',
    [ValidateSet('cpu','cuda')][string]$Backend = 'cuda',
    [string[]]$Cases = @('upsample8','upsample10'),
    [ValidateRange(3,40)][int]$Runs = 12,
    [ValidateRange(1,40)][int]$NcuLaunchCount = 8,
    [string]$PythonExe = 'python',
    [switch]$NoTorch,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$probe = Join-Path $root 'stage111_phase_gemm_probe.exe'
$validCases = @('small','upsample8','upsample10','upsample4')
$validModes = @('direct','phase_f32','phase_f16')
foreach ($case in $Cases) {
    if ($case -notin $validCases) { throw "Unsupported case: $case" }
}
if ($DryRun) {
    foreach ($case in $Cases) {
        foreach ($mode in $validModes) { Write-Host "$case $mode" }
    }
    Write-Host "Stage111 profiler=$Profiler backend=$Backend torch=$(-not $NoTorch)"
    return
}
if (-not (Test-Path -LiteralPath $probe -PathType Leaf)) { throw "Missing probe: $probe" }
if ($Profiler -eq 'ncu' -and $Backend -ne 'cuda') { throw 'NCU requires CUDA backend' }
$work = Join-Path $root ('stage111_results\' + [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
New-Item -ItemType Directory -Force -Path $work | Out-Null
$env:RVC_NVTX = '1'
if ($Profiler -eq 'ncu') {
    $ncu = (Get-Command ncu -ErrorAction Stop).Source
    $torchFile = & $PythonExe -c 'import torch; print(torch.__file__)'
    if ($LASTEXITCODE -ne 0 -or -not $torchFile) { throw 'Torch needed to locate NVTX DLL' }
    $torchLib = Join-Path (Split-Path -Parent ((@($torchFile) | Select-Object -Last 1).Trim())) 'lib'
    $nvtxDll = Join-Path $torchLib 'nvToolsExt64_1.dll'
    if (-not (Test-Path -LiteralPath $nvtxDll -PathType Leaf)) { throw "NVTX DLL not found: $nvtxDll" }
    $env:PATH = "$torchLib;$env:PATH"
    $status = & $probe nvtx-status
    if ($LASTEXITCODE -ne 0 -or @($status) -notcontains 'nvtx_available=true') {
        throw "Stage111 NVTX not available: $status"
    }
}
$records = @()
foreach ($case in $Cases) {
    foreach ($mode in $validModes) {
        $stem = "$case-$mode"
        $log = Join-Path $work "$stem.benchmark.log"
        Write-Host "Stage111 benchmark $stem"
        & $probe $Backend $case $mode $Runs 2>&1 | Tee-Object -FilePath $log | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Stage111 benchmark failed: $stem, log $log" }
        $jsonLine = Get-Content -LiteralPath $log | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
        if (-not $jsonLine) { throw "Benchmark JSON missing: $stem" }
        $data = $jsonLine | ConvertFrom-Json
        if ($data.full_output_parity -ne $true) { throw "Numerical parity failed: $stem" }
        $jsonLine | Set-Content -LiteralPath (Join-Path $work "$stem.benchmark.json") -Encoding utf8
        if ($Profiler -eq 'ncu') {
            $path = Join-Path $work "$stem.ncu"
            $ncuLog = Join-Path $work "$stem.ncu.log"
            $nvtxRange = "stage111_$($case)_$($mode)/"
            $argsNcu = @(
                '--target-processes','all',
                '--nvtx','--nvtx-push-pop-scope','process',
                '--nvtx-include',$nvtxRange,
                '--launch-count',"$NcuLaunchCount",
                '--section','SpeedOfLight',
                '--section','LaunchStats',
                '--section','Occupancy',
                '--section','MemoryWorkloadAnalysis',
                '--section','SchedulerStats',
                '--section','WarpStateStats',
                '--export',$path,'--force-overwrite',
                $probe,$Backend,$case,$mode,'4'
            )
            Write-Host "Stage111 NCU capture $stem"
            & $ncu @argsNcu 2>&1 | Tee-Object -FilePath $ncuLog | Out-Host
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath "$path.ncu-rep" -PathType Leaf)) {
                throw "Stage111 NCU failed: $stem. Log: $ncuLog"
            }
        }
        $records += [pscustomobject]@{
            case=$case;mode=$mode;median_ms=$data.median_ms;
            max_abs=$data.max_abs;snr_db=$data.snr_db;dtype=$data.dtype
        }
    }
    if (-not $NoTorch -and $Backend -eq 'cuda') {
        foreach ($dtype in @('f32','f16','f16_bridge')) {
            $torchLog = Join-Path $work "torch-$case-$dtype.log"
            Write-Host "Stage111 PyTorch/cuDNN benchmark $case/$dtype"
            & $PythonExe (Join-Path $root 'torch_reference.py') $case $dtype --runs $Runs 2>&1 | Tee-Object -FilePath $torchLog | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "Stage111 torch benchmark failed: $case/$dtype" }
            $torchJson = Get-Content -LiteralPath $torchLog | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
            if (-not $torchJson) { throw "Stage111 Torch JSON missing: $case/$dtype" }
            $torchJson | Set-Content -LiteralPath (Join-Path $work "torch-$case-$dtype.json") -Encoding utf8
        }
    }
}
$records | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $work 'manifest.json') -Encoding utf8
& $PythonExe (Join-Path $root 'compare_stage111.py') $work ($Cases -join ',')
if ($LASTEXITCODE -ne 0) { throw 'Stage111 comparator failed' }
$archive = Join-Path $root 'RVC_STAGE111_PHASE_GEMM_RESULTS.zip'
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path (Join-Path $work '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Stage111 completed: $archive"
