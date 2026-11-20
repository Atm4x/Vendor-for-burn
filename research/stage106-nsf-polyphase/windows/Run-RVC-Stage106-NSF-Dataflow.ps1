param(
    [ValidateSet('none','ncu')][string]$Profiler = 'none',
    [ValidateSet('cpu','cuda')][string]$Backend = 'cuda',
    [string[]]$TransposeCases = @('upsample8', 'upsample10'),
    [string[]]$NoiseCases = @('noise8', 'noise40'),
    [ValidateRange(2,40)][int]$Runs = 9,
    [ValidateRange(1,50)][int]$NcuLaunchCount = 3,
    [string]$PythonExe = 'python',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$transpose = Join-Path $root 'stage106_polyphase_probe.exe'
$noise = Join-Path $root 'stage106_noise_probe.exe'
$stamp = [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$work = Join-Path $root "stage106_results\$stamp"
$planned = @()
foreach ($case in $TransposeCases) {
    foreach ($mode in @('direct','polyphase')) {
        $planned += [pscustomobject]@{exe=$transpose;case=$case;mode=$mode;op='transpose'}
    }
}
foreach ($case in $NoiseCases) {
    foreach ($mode in @('direct','tiled')) {
        $planned += [pscustomobject]@{exe=$noise;case=$case;mode=$mode;op='noise'}
    }
}
if ($DryRun) {
    $planned | Select-Object op,case,mode | Format-Table | Out-String | Write-Host
    Write-Host "Stage106: $($planned.Count) targeted captures, profiler=$Profiler, backend=$Backend"
    exit 0
}
if ($Profiler -eq 'ncu' -and $Backend -ne 'cuda') {throw 'NCU requires -Backend cuda'}
foreach ($exe in @($transpose,$noise)) {
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {throw "Missing executable: $exe"}
}
New-Item -ItemType Directory -Force -Path $work | Out-Null
$env:RVC_NVTX = '1'
if ($Profiler -eq 'ncu') {
    $ncu = (Get-Command ncu -ErrorAction Stop).Source
    $py = & $PythonExe -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $py -PathType Container)) {
        $env:PATH = "$py;$env:PATH"
    }
}
$records = @()
foreach ($item in $planned) {
    $stem = "$($item.op)-$($item.case)-$($item.mode)"
    $log = Join-Path $work "$stem.benchmark.log"
    Write-Host "Stage106 benchmark $stem"
    & $item.exe $Backend $item.case $item.mode $Runs 2>&1 | Tee-Object -FilePath $log | Out-Host
    if ($LASTEXITCODE -ne 0) {throw "Benchmark failed: $stem. See $log"}
    $lines = Get-Content -LiteralPath $log
    $jsonLine = $lines | Where-Object { $_.TrimStart().StartsWith('{') } | Select-Object -Last 1
    if (-not $jsonLine) {throw "Missing benchmark JSON: $stem"}
    $benchmark = $jsonLine | ConvertFrom-Json
    if ($benchmark.full_output_parity -ne $true) {throw "Numerical parity failed: $stem"}
    $jsonLine | Set-Content -LiteralPath (Join-Path $work "$stem.benchmark.json") -Encoding utf8
    $status = 'not_requested'
    $report = $null
    if ($Profiler -eq 'ncu') {
        $range = if ($item.op -eq 'transpose') {
            "stage106_$($item.case)_$($item.mode)/"
        } else {
            "stage106_noise_$($item.case)_$($item.mode)/"
        }
        $stemPath = Join-Path $work "$stem.ncu"
        $ncuLog = Join-Path $work "$stem.ncu.log"
        $ncuArgs = @(
            '--target-processes','all',
            '--nvtx','--nvtx-push-pop-scope','process',
            '--nvtx-include',$range,
            '--launch-count',"$NcuLaunchCount",
            '--section','SpeedOfLight',
            '--section','LaunchStats',
            '--section','Occupancy',
            '--section','MemoryWorkloadAnalysis',
            '--section','SchedulerStats',
            '--section','WarpStateStats',
            '--export',$stemPath,
            '--force-overwrite',
            $item.exe,$Backend,$item.case,$item.mode,'4'
        )
        Write-Host "Stage106 NCU capture $stem"
        & $ncu @ncuArgs 2>&1 | Tee-Object -FilePath $ncuLog | Out-Host
        $report = "$stemPath.ncu-rep"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $report -PathType Leaf)) {
            throw "NCU capture failed: $stem (see $ncuLog). Check NVTX and ERR_NVGPUCTRPERM"
        }
        $status = 'captured'
    }
    $records += [pscustomobject]@{
        op=$item.op;case=$item.case;mode=$item.mode;median_ms=$benchmark.median_ms;
        max_abs=$benchmark.max_abs;snr_db=$benchmark.snr_db;
        full_output_parity=$benchmark.full_output_parity;ncu=$status
    }
}
$records | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $work 'manifest.json') -Encoding utf8
& $PythonExe (Join-Path $root 'compare_stage106.py') $work
if ($LASTEXITCODE -ne 0) {throw 'Stage106 comparison analyzer failed'}
$results = Join-Path $root 'RVC_STAGE106_NSF_DATAFLOW_RESULTS.zip'
if (Test-Path -LiteralPath $results) {Remove-Item -LiteralPath $results -Force}
Compress-Archive -Path (Join-Path $work '*') -DestinationPath $results -CompressionLevel Optimal
Write-Host "Stage106 all requested checks completed. Archive: $results"
