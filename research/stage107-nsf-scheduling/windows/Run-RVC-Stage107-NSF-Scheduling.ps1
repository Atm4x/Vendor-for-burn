param(
    [ValidateSet('none','ncu')][string]$Profiler = 'none',
    [ValidateSet('cpu','cuda')][string]$Backend = 'cuda',
    [string[]]$TransposeCases = @('upsample8', 'upsample10'),
    [string[]]$NoiseCases = @('noise8', 'noise40'),
    [ValidateRange(2,40)][int]$Runs = 12,
    [ValidateRange(1,50)][int]$NcuLaunchCount = 3,
    [string]$PythonExe = 'python',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$transpose = Join-Path $root 'stage106_polyphase_probe.exe'
$noise = Join-Path $root 'stage106_noise_probe.exe'
$stamp = [DateTimeOffset]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$work = Join-Path $root "stage107_results\$stamp"
$planned = @()

foreach ($case in $TransposeCases) {
    foreach ($mode in @('direct','polyphase','single')) {
        $planned += [pscustomobject]@{ exe=$transpose; case=$case; mode=$mode; op='transpose' }
    }
}
foreach ($case in $NoiseCases) {
    foreach ($mode in @('direct','tiled','interior')) {
        $planned += [pscustomobject]@{ exe=$noise; case=$case; mode=$mode; op='noise' }
    }
}

if ($DryRun) {
    $planned | Select-Object op,case,mode | Format-Table | Out-String | Write-Host
    Write-Host "Stage107: $($planned.Count) runs; profiler=$Profiler; backend=$Backend"
    exit 0
}

if ($Profiler -eq 'ncu' -and $Backend -ne 'cuda') {
    throw 'NCU requires CUDA backend'
}
foreach ($exe in @($transpose,$noise)) {
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        throw "Missing executable: $exe"
    }
}
New-Item -ItemType Directory -Force -Path $work | Out-Null
$env:RVC_NVTX = '1'

if ($Profiler -eq 'ncu') {
    $ncu = (Get-Command ncu -ErrorAction Stop).Source
    $torchModule = & $PythonExe -c 'import torch; print(torch.__file__)'
    if ($LASTEXITCODE -ne 0 -or -not $torchModule) {
        throw 'Could not resolve PyTorch installation for nvToolsExt64_1.dll'
    }
    $torchModule = (@($torchModule) | Select-Object -Last 1).Trim()
    $torchLib = Join-Path (Split-Path -Parent $torchModule) 'lib'
    $nvtxDll = Join-Path $torchLib 'nvToolsExt64_1.dll'
    if (-not (Test-Path -LiteralPath $nvtxDll -PathType Leaf)) {
        throw "NVTX DLL missing: $nvtxDll"
    }
    $env:PATH = "$torchLib;$env:PATH"
    foreach ($exe in @($transpose,$noise)) {
        $nvtx = & $exe nvtx-status
        if ($LASTEXITCODE -ne 0 -or @($nvtx) -notcontains 'nvtx_available=true') {
            throw "NVTX initialization failed in ${exe}: $nvtx"
        }
    }
}

$records = @()
foreach ($item in $planned) {
    $stem = "$($item.op)-$($item.case)-$($item.mode)"
    $log = Join-Path $work "$stem.benchmark.log"
    Write-Host "Stage107 benchmark $stem"
    & $item.exe $Backend $item.case $item.mode $Runs 2>&1 | Tee-Object -FilePath $log | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark failed: $stem. See $log"
    }
    $line = Get-Content -LiteralPath $log |
        Where-Object { $_.TrimStart().StartsWith('{') } |
        Select-Object -Last 1
    if (-not $line) { throw "Missing benchmark JSON: $stem" }
    $benchmark = $line | ConvertFrom-Json
    if ($benchmark.full_output_parity -ne $true -or $benchmark.max_abs -gt 1e-4) {
        throw "Full-output numerical parity failed: $stem"
    }
    $line | Set-Content -LiteralPath (Join-Path $work "$stem.benchmark.json") -Encoding utf8
    $status = 'not_requested'
    if ($Profiler -eq 'ncu') {
        $range = if ($item.op -eq 'transpose') {
            "stage106_$($item.case)_$($item.mode)/"
        } else {
            "stage106_noise_$($item.case)_$($item.mode)/"
        }
        $reportBase = Join-Path $work "$stem.ncu"
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
            '--export',$reportBase,
            '--force-overwrite',
            $item.exe,$Backend,$item.case,$item.mode,'4'
        )
        Write-Host "Stage107 NCU $stem"
        & $ncu @ncuArgs 2>&1 | Tee-Object -FilePath $ncuLog | Out-Host
        $report = "$reportBase.ncu-rep"
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $report -PathType Leaf)) {
            throw "NCU failed for $stem. See $ncuLog"
        }
        if ((Get-Item -LiteralPath $report).Length -lt 1024) {
            throw "NCU report too small: $report"
        }
        if (Select-String -Path $ncuLog -Pattern 'No kernels were profiled' -Quiet) {
            throw "NCU range matched no kernels: $stem"
        }
        $status = 'captured'
    }
    $records += [pscustomobject]@{
        op=$item.op; case=$item.case; mode=$item.mode;
        median_ms=$benchmark.median_ms; max_abs=$benchmark.max_abs;
        snr_db=$benchmark.snr_db; full_output_parity=$true; ncu=$status
    }
}
$records | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $work 'manifest.json') -Encoding utf8
& $PythonExe (Join-Path $root 'compare_stage107.py') $work
if ($LASTEXITCODE -ne 0) {throw 'Stage107 comparison analyzer failed'}
$zip = Join-Path $root 'RVC_STAGE107_NSF_SCHEDULING_RESULTS.zip'
if (Test-Path -LiteralPath $zip) {Remove-Item -LiteralPath $zip -Force}
Compress-Archive -Path (Join-Path $work '*') -DestinationPath $zip -CompressionLevel Optimal
Write-Host "Stage107 verified results: $zip"
