param(
    [string]$Probe = '.\stage103-conv2d-probe.exe',
    [string]$PythonExe = 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe',
    [ValidateRange(2,64)][int]$Runs = 12,
    [ValidateSet('none','ncu')][string]$Profiler = 'ncu',
    [ValidateRange(0,3)][int]$NcuSkip = 0,
    [string]$NcuExe = 'ncu'
)
$ErrorActionPreference = 'Stop'
$reports = Join-Path $PSScriptRoot 'stage103_results'
New-Item -ItemType Directory -Force -Path $reports | Out-Null
$runner = (Resolve-Path -LiteralPath $Probe).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$compare = Join-Path $PSScriptRoot 'compare_stage103.py'
if (-not (Test-Path $compare)) { throw 'Missing comparison script' }
if ($Profiler -eq 'ncu' -and -not (Get-Command $NcuExe -ErrorAction SilentlyContinue)) { throw 'ncu not found' }
$oldNvtx = $env:RVC_NVTX
$oldMode = $env:RVC_STAGE103_STRATEGY
$oldPath = $env:PATH
$completed = $false
function Invoke-Logged([string]$Command, [string[]]$Arguments, [string]$Log) {
    Write-Host "Running: $(Split-Path $Command -Leaf) $($Arguments -join ' ')"
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command @Arguments *>&1 | Out-File -LiteralPath $Log -Encoding UTF8
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $old
    }
    if ($code -ne 0) {
        Get-Content -LiteralPath $Log -Tail 45 | ForEach-Object { Write-Host $_ }
        throw "Command failed with exit code $code; log: $Log"
    }
}
try {
    $env:RVC_NVTX = if ($Profiler -eq 'ncu') { '1' } else { '0' }
    $venv = Split-Path (Split-Path $python -Parent) -Parent
    $nvtxDll = Join-Path $venv 'Lib\site-packages\torch\lib\nvToolsExt64_1.dll'
    if (Test-Path -LiteralPath $nvtxDll) {
        $env:PATH = "$(Split-Path $nvtxDll -Parent);$env:PATH"
    }
    if ($Profiler -eq 'ncu') {
        $nvtx = & $runner nvtx-status
        if ($LASTEXITCODE -ne 0 -or $nvtx -ne 'nvtx_available=true') {
            throw 'NVTX unavailable. Check nvToolsExt64_1.dll and PATH.'
        }
    }
    foreach ($mode in @('direct','reuse')) {
        $env:RVC_STAGE103_STRATEGY = $mode
        Invoke-Logged $runner @('cuda',"$Runs") (Join-Path $reports "$mode.json")
    }
    Invoke-Logged $python @($compare,'--direct',(Join-Path $reports 'direct.json'),
        '--reuse',(Join-Path $reports 'reuse.json'),
        '--output',(Join-Path $reports 'comparison.json')) (Join-Path $reports 'comparison.log')
    Get-Content (Join-Path $reports 'comparison.log') | Write-Host
    if ($Profiler -eq 'ncu') {
        foreach ($mode in @('direct','reuse')) {
            $env:RVC_STAGE103_STRATEGY = $mode
            $filter = if ($mode -eq 'direct') { 'regex:direct_conv2d_kernel' }
                      else { 'regex:spatial_reuse_3x3_kernel' }
            $output = Join-Path $reports "$mode-ncu"
            $flags = @(
                '--target-processes','all', '--nvtx','--nvtx-push-pop-scope','process',
                '--nvtx-include','stage102_conv2d_matched/', '--kernel-name',$filter,
                '--launch-skip',"$NcuSkip",'--launch-count','1',
                '--section','SpeedOfLight','--section','LaunchStats',
                '--section','Occupancy','--section','MemoryWorkloadAnalysis',
                '--export',$output,'--force-overwrite',$runner,'cuda','3'
            )
            Invoke-Logged $NcuExe $flags (Join-Path $reports "$mode-ncu.log")
            if (-not (Test-Path -LiteralPath "$output.ncu-rep")) {
                Get-Content (Join-Path $reports "$mode-ncu.log") -Tail 40 | Write-Host
                throw "No Nsight Compute report: $output.ncu-rep"
            }
            Write-Host "NCU report: $output.ncu-rep"
        }
    }
    [pscustomobject]@{
        stage=103; modes=@('direct','reuse'); precision='fp32'; layout='nchw'; runs=$Runs;
        profiler=$Profiler; ncu_skip=$NcuSkip; nvtx='stage102_conv2d_matched';
        output_audio_in_zip=$false; full_tensor_parity_required=$true;
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reports 'manifest.json') -Encoding UTF8
    $completed = $true
} finally {
    $env:RVC_NVTX = $oldNvtx
    $env:RVC_STAGE103_STRATEGY = $oldMode
    $env:PATH = $oldPath
    if ($completed) {
        $archive = Join-Path $PSScriptRoot 'RVC_STAGE103_L1_REUSE_RESULTS.zip'
        if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
        Compress-Archive -Path (Join-Path $reports '*') -DestinationPath $archive -CompressionLevel Optimal
        Write-Host "Stage103 result ZIP (no WAV): $archive"
    } else {
        Write-Warning "Stage103 incomplete. Logs remain in $reports; no success ZIP created."
    }
}
