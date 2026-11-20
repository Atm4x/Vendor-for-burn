param(
    [string]$RustProbe = '.\stage102-conv2d-probe.exe',
    [string]$PythonExe = 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe',
    [ValidateRange(2,64)][int]$Runs = 12,
    [ValidateRange(1,8)][int]$NcuLaunchCount = 3,
    [ValidateSet('fp32','fp16')][string]$Precision = 'fp32',
    [ValidateSet('nchw','nhwc')][string]$Layout = 'nchw',
    [ValidateSet('none','ncu','nsys','both')][string]$Profiler = 'both',
    [string]$NsysExe = 'nsys',
    [string]$NcuExe = 'ncu'
)
$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot 'stage102_matched_work'
$reports = Join-Path $root 'reports'
New-Item -ItemType Directory -Force -Path $reports | Out-Null
$torchProbe = Join-Path $PSScriptRoot 'torch_stage102_conv2d.py'
$comparator = Join-Path $PSScriptRoot 'compare_stage102_conv2d.py'
foreach ($file in @($RustProbe,$PythonExe,$torchProbe,$comparator)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
if ($Profiler -in @('both','nsys') -and -not (Get-Command $NsysExe -ErrorAction SilentlyContinue)) { throw 'nsys not found' }
if ($Profiler -in @('both','ncu') -and -not (Get-Command $NcuExe -ErrorAction SilentlyContinue)) { throw 'ncu not found' }
$rust = (Resolve-Path -LiteralPath $RustProbe).Path
$python = (Resolve-Path -LiteralPath $PythonExe).Path
$oldNvtx = [Environment]::GetEnvironmentVariable('RVC_NVTX','Process')
$oldVisible = [Environment]::GetEnvironmentVariable('CUDA_VISIBLE_DEVICES','Process')
function Invoke-Captured([string]$Command, [string[]]$Arguments, [string]$Log) {
    Write-Host "Running: $(Split-Path $Command -Leaf) $($Arguments -join ' ')"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command @Arguments *>&1 | Out-File -LiteralPath $Log -Encoding UTF8
        $exitCode = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previous }
    if ($exitCode -ne 0) {
        Get-Content -LiteralPath $Log -Tail 35 | ForEach-Object { Write-Host $_ }
        throw "Command failed exit=$exitCode, log=$Log"
    }
}
try {
    $env:RVC_NVTX = if ($Profiler -eq 'none') { '0' } else { '1' }
    $rustNvtx = & $rust nvtx-status 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'Stage102 CUDA executable does not support NVTX status' }
    if ($Profiler -ne 'none' -and $rustNvtx -ne 'nvtx_available=true') {
        throw 'NVTX runtime missing. Make nvToolsExt64_1.dll available on PATH before profiling.'
    }
    $env:CUDA_VISIBLE_DEVICES = '0'
    $torchCmd = @($torchProbe,'--backend','cuda','--dtype',$Precision,'--layout',$Layout,'--repeats',"$Runs",
        '--output',(Join-Path $reports 'torch.json'))
    $rustCmd = @('cuda',"$Runs")
    Invoke-Captured $rust $rustCmd (Join-Path $reports 'rust.json')
    Invoke-Captured $python $torchCmd (Join-Path $reports 'torch.log')
    if ($Precision -eq 'fp32' -and $Layout -eq 'nchw') {
        Invoke-Captured $python @($comparator,'--rust',(Join-Path $reports 'rust.json'),
            '--torch',(Join-Path $reports 'torch.json'),'--output',(Join-Path $reports 'matched-comparison.json')) (Join-Path $reports 'compare.log')
    } else {
        'Only FP32 NCHW is strictly matched with Burn. Precision/layout variants are diagnostic, not parity results.' |
            Set-Content -LiteralPath (Join-Path $reports 'unmatched.txt') -Encoding UTF8
    }
    if ($Profiler -ne 'none') {
        $targets = @(
            [pscustomobject]@{Name='rust';Exe=$rust;Args=$rustCmd;Filter='direct_conv2d_kernel';Range='stage102_conv2d_matched/'},
            [pscustomobject]@{Name='torch';Exe=$python;Args=$torchCmd;Filter='cudnn|cutlass|xmma';Range='stage102_conv2d_matched/'}
        )
        foreach ($target in $targets) {
            if ($Profiler -in @('both','nsys')) {
                Write-Host "NSYS matched Conv2d: $($target.Name)"
                Invoke-Captured $NsysExe (@('profile','--trace=cuda,nvtx','--sample=none','--force-overwrite=true',
                    '--output',(Join-Path $reports "$($target.Name)-nsys"),$target.Exe) + $target.Args) (Join-Path $reports "$($target.Name)-nsys.log")
            }
            if ($Profiler -in @('both','ncu')) {
                Write-Host "NCU matched Conv2d: $($target.Name)"
                $output = Join-Path $reports "$($target.Name)-ncu"
                $flags = @('--target-processes','all','--nvtx','--nvtx-push-pop-scope','process','--nvtx-include',$target.Range,
                    '--kernel-name',"regex:$($target.Filter)",'--launch-count',"$NcuLaunchCount",
                    '--section','SpeedOfLight','--section','LaunchStats','--section','Occupancy',
                    '--section','MemoryWorkloadAnalysis','--export',$output,'--force-overwrite',$target.Exe) + $target.Args
                Invoke-Captured $NcuExe $flags (Join-Path $reports "$($target.Name)-ncu.log")
                $rep = "$output.ncu-rep"
                if (-not (Test-Path -LiteralPath $rep)) { throw "No NCU report: $rep" }
                Write-Host "NCU report saved: $rep (open in Nsight Compute GUI; CLI export may fail on this version)"
            }
        }
    }
    [pscustomobject]@{stage=102; shape_contract='identical'; precision_rust='fp32'; precision_torch=$Precision;
        layout_rust='nchw'; layout_torch=$Layout; runs=$Runs; nvtx='stage102_conv2d_matched'; profiler=$Profiler;
        output_audio_in_zip=$false} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $reports 'manifest.json') -Encoding UTF8
    $completed = $true
} finally {
    [Environment]::SetEnvironmentVariable('RVC_NVTX',$oldNvtx,'Process')
    [Environment]::SetEnvironmentVariable('CUDA_VISIBLE_DEVICES',$oldVisible,'Process')
    if ($completed) {
        $archive = Join-Path $PSScriptRoot 'RVC_STAGE102_MATCHED_CONV2D_RESULTS.zip'
        if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
        Compress-Archive -Path (Join-Path $reports '*') -DestinationPath $archive -CompressionLevel Optimal
        Write-Host "Stage102 matched Conv2d ZIP (no audio): $archive"
    } else {
        Write-Warning "Stage102 failed; incomplete reports kept in $reports"
    }
}
