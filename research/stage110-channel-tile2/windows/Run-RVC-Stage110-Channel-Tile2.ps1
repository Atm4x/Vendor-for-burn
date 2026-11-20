param(
    [ValidateSet('none','ncu')][string]$Profiler = 'ncu',
    [ValidateSet('cpu','cuda')][string]$Backend = 'cuda',
    [string[]]$Cases = @('upsample8','upsample10'),
    [string[]]$Modes = @('single','ilp4','tile2','tile2_split'),
    [ValidateRange(3,40)][int]$Runs = 12,
    [ValidateRange(1,20)][int]$NcuLaunchCount = 2,
    [string]$PythonExe = 'python',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$probe = Join-Path $root 'stage106_polyphase_probe.exe'
$work = Join-Path $root ('stage110_results\' + [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ'))
$allowed = @('single','ilp4','tile2','tile2_split')
$plan = @()
foreach ($case in $Cases) {
    foreach ($mode in $Modes) {
        if ($mode -notin $allowed) {throw "Unsupported Stage110 mode: $mode"}
        $plan += [pscustomobject]@{case=$case;mode=$mode}
    }
}
if ($Modes[0] -ne 'single') {throw 'Stage110 requires single baseline as first mode'}
if ($DryRun) {
    $plan | Format-Table | Out-String | Write-Host
    Write-Host "Stage110: $($plan.Count) benchmark runs; profiler=$Profiler; backend=$Backend"
    return
}
if (-not (Test-Path -LiteralPath $probe -PathType Leaf)) {throw "Missing probe: $probe"}
if ($Profiler -eq 'ncu' -and $Backend -ne 'cuda') {throw 'NCU requires CUDA backend'}
New-Item -ItemType Directory -Force -Path $work | Out-Null
$env:RVC_NVTX = '1'
if ($Profiler -eq 'ncu') {
    $ncu = (Get-Command ncu -ErrorAction Stop).Source
    $torchModule = & $PythonExe -c 'import torch; print(torch.__file__)'
    if ($LASTEXITCODE -ne 0 -or -not $torchModule) {throw 'PyTorch not found for NVTX DLL'}
    $torchLib = Join-Path (Split-Path -Parent ((@($torchModule) | Select-Object -Last 1).Trim())) 'lib'
    $dll = Join-Path $torchLib 'nvToolsExt64_1.dll'
    if (-not (Test-Path -LiteralPath $dll -PathType Leaf)) {throw "Missing NVTX DLL: $dll"}
    $env:PATH = "$torchLib;$env:PATH"
    $nvtxStatus = & $probe nvtx-status
    if ($LASTEXITCODE -ne 0 -or @($nvtxStatus) -notcontains 'nvtx_available=true') {
        throw "NVTX unavailable in Stage110 probe: $nvtxStatus"
    }
}
foreach ($item in $plan) {
    $stem = "transpose-$($item.case)-$($item.mode)"
    $log = Join-Path $work "$stem.benchmark.log"
    Write-Host "Stage110 benchmark $stem"
    & $probe $Backend $item.case $item.mode $Runs 2>&1 | Tee-Object -FilePath $log | Out-Host
    if ($LASTEXITCODE -ne 0) {throw "Stage110 benchmark failed: $stem"}
    $line = Get-Content -LiteralPath $log |
        Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
    if (-not $line) {throw "Missing benchmark JSON: $stem"}
    $data = $line | ConvertFrom-Json
    if ($data.full_output_parity -ne $true -or $data.max_abs -gt 1e-4) {
        throw "Numerical parity failed: $stem"
    }
    $line | Set-Content -LiteralPath (Join-Path $work "$stem.benchmark.json") -Encoding utf8
    if ($Profiler -eq 'ncu') {
        $stemPath = Join-Path $work "$stem.ncu"
        $ncuLog = Join-Path $work "$stem.ncu.log"
        $nvtxRange = "stage106_$($item.case)_$($item.mode)/"
        $arguments = @(
            '--target-processes','all',
            '--nvtx','--nvtx-push-pop-scope','process','--nvtx-include',$nvtxRange,
            '--kernel-name','regex:polyphase_',
            '--launch-count',"$NcuLaunchCount",
            '--section','SpeedOfLight','--section','LaunchStats',
            '--section','Occupancy','--section','MemoryWorkloadAnalysis',
            '--section','SchedulerStats','--section','WarpStateStats',
            '--export',$stemPath,'--force-overwrite',$probe,$Backend,$item.case,$item.mode,'4'
        )
        Write-Host "Stage110 NCU capture $stem"
        & $ncu @arguments 2>&1 | Tee-Object -FilePath $ncuLog | Out-Host
        $report = "$stemPath.ncu-rep"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $report -PathType Leaf)) {
            throw "NCU capture failed: $stem. Inspect $ncuLog"
        }
    }
}
& $PythonExe (Join-Path $root 'compare_stage110.py') $work ($Cases -join ',') ($Modes -join ',')
if ($LASTEXITCODE -ne 0) {throw 'Stage110 analyzer failed'}
$archive = Join-Path $root 'RVC_STAGE110_CHANNEL_TILE2_RESULTS.zip'
if (Test-Path -LiteralPath $archive) {Remove-Item -LiteralPath $archive -Force}
Compress-Archive -Path (Join-Path $work '*') -DestinationPath $archive -CompressionLevel Optimal
Write-Host "Stage110 verified: $archive"
