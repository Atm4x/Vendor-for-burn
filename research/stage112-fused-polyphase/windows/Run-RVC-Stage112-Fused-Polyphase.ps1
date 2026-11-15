param(
    [ValidateSet('none','ncu')][string]$Profiler = 'none',
    [ValidateSet('cpu','cuda')][string]$Backend = 'cuda',
    [string[]]$Cases = @('upsample8','upsample10'),
    [string[]]$Modes = @('direct','ilp4','fused2tap'),
    [ValidateRange(3,40)][int]$Runs = 12,
    [ValidateRange(1,20)][int]$NcuLaunchCount = 2,
    [string]$PythonExe = 'python',
    [switch]$NoTorch,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$probe = Join-Path $root 'stage112_fused_probe.exe'
$validCases = @('upsample8','upsample10','upsample4','upsample2','grouped2','odd2')
$validModes = @('direct','ilp4','fused2tap')
foreach ($case in $Cases) { if ($case -notin $validCases) { throw "Unsupported two-tap case: $case" } }
foreach ($mode in $Modes) { if ($mode -notin $validModes) { throw "Unsupported mode: $mode" } }
if ($Modes -notcontains 'fused2tap') { throw 'Include fused2tap in benchmark modes' }
if ($DryRun) {
    foreach ($case in $Cases) { foreach ($mode in $Modes) { Write-Host "Stage112 plan $case $mode" } }
    return
}
if (-not (Test-Path -LiteralPath $probe -PathType Leaf)) { throw "Missing probe: $probe" }
if ($Profiler -eq 'ncu' -and $Backend -ne 'cuda') { throw 'NCU requires CUDA' }
$work = Join-Path $root ('stage112_results\' + [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
New-Item -ItemType Directory -Force -Path $work | Out-Null
$env:RVC_NVTX = '1'
if ($Profiler -eq 'ncu') {
    $ncu = (Get-Command ncu -ErrorAction Stop).Source
    $torchModule = & $PythonExe -c 'import torch; print(torch.__file__)'
    if ($LASTEXITCODE -ne 0 -or -not $torchModule) { throw 'PyTorch is required to locate NVTX DLL' }
    $torchLib = Join-Path (Split-Path -Parent ((@($torchModule) | Select-Object -Last 1).Trim())) 'lib'
    $nvtx = Join-Path $torchLib 'nvToolsExt64_1.dll'
    if (-not (Test-Path -LiteralPath $nvtx -PathType Leaf)) { throw "Missing NVTX DLL: $nvtx" }
    $env:PATH = "$torchLib;$env:PATH"
    $status = & $probe nvtx-status
    if ($LASTEXITCODE -ne 0 -or @($status) -notcontains 'nvtx_available=true') {
        throw "Stage112 NVTX initialization failed: $status"
    }
    foreach ($case in $Cases) {
        foreach ($mode in $Modes) {
            Write-Host "Stage112 preflight CUDA parity $case/$mode"
            & $probe $Backend $case $mode 3 2>&1 | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "Stage112 CUDA parity failed before NCU: $case/$mode" }
        }
    }
}
foreach ($case in $Cases) {
    foreach ($mode in $Modes) {
        $stem = "$case-$mode"
        $log = Join-Path $work "$stem.benchmark.log"
        Write-Host "Stage112 benchmark $stem"
        & $probe $Backend $case $mode $Runs 2>&1 | Tee-Object -FilePath $log | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Stage112 benchmark failed: $stem" }
        $line = Get-Content -LiteralPath $log | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
        if (-not $line) { throw "Stage112 benchmark output missing: $stem" }
        $data = $line | ConvertFrom-Json
        if ($data.full_output_parity -ne $true -or $data.max_abs -gt 1e-4) { throw "Stage112 parity failed: $stem" }
        $line | Set-Content -LiteralPath (Join-Path $work "$stem.benchmark.json") -Encoding utf8
        if ($Profiler -eq 'ncu') {
            $path = Join-Path $work "$stem.ncu"
            $logNcu = Join-Path $work "$stem.ncu.log"
            $nvtxRange = "stage112_$($case)_$($mode)/"
            $filter = switch ($mode) {
                'direct' { 'regex:conv_transpose2d_direct' }
                'ilp4' { 'regex:polyphase_pipeline4' }
                'fused2tap' { 'regex:fused_two_tap' }
            }
            $ncuArgs = @(
                '--target-processes','all','--nvtx',
                '--nvtx-push-pop-scope','process','--nvtx-include',$nvtxRange,
                '--kernel-name',$filter,'--launch-count',"$NcuLaunchCount",
                '--section','SpeedOfLight','--section','LaunchStats',
                '--section','Occupancy','--section','MemoryWorkloadAnalysis',
                '--section','SchedulerStats','--section','WarpStateStats',
                '--export',$path,'--force-overwrite',$probe,$Backend,$case,$mode,'4'
            )
            Write-Host "Stage112 NCU capture $stem"
            & $ncu @ncuArgs 2>&1 | Tee-Object -FilePath $logNcu | Out-Host
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath "$path.ncu-rep" -PathType Leaf)) {
                throw "Stage112 NCU failed: $stem. Inspect $logNcu"
            }
        }
    }
    if (-not $NoTorch -and $Backend -eq 'cuda' -and $case -in @('upsample8','upsample10','upsample4')) {
        foreach ($dtype in @('f32','f16','f16_bridge')) {
            $torchLog = Join-Path $work "torch-$case-$dtype.log"
            Write-Host "Stage112 matched PyTorch $case/$dtype"
            & $PythonExe (Join-Path $root 'torch_reference_stage112.py') $case $dtype --runs $Runs 2>&1 |
                Tee-Object -FilePath $torchLog | Out-Host
            if ($LASTEXITCODE -ne 0) { throw "PyTorch benchmark failed: $case/$dtype" }
            $torchLine = Get-Content -LiteralPath $torchLog | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
            if (-not $torchLine) { throw "PyTorch output missing: $case/$dtype" }
            $torchLine | Set-Content -LiteralPath (Join-Path $work "torch-$case-$dtype.json") -Encoding utf8
        }
    }
}
& $PythonExe (Join-Path $root 'compare_stage112.py') $work ($Cases -join ',') ($Modes -join ',')
if ($LASTEXITCODE -ne 0) { throw 'Stage112 result validation failed' }
$archive = Join-Path $root 'RVC_STAGE112_FUSED_POLYPHASE_RESULTS.zip'
if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
Compress-Archive -Path (Join-Path $work '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Stage112 verified: $archive"
