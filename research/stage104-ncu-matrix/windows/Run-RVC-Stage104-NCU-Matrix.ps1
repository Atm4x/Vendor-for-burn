param(
    [ValidateSet('micro','pipeline','all')][string]$Scope = 'micro',
    [ValidateSet('balanced','full')][string]$MetricSet = 'balanced',
    [string]$Cases = 'dense3x3,dense3x3_wide,pointwise1x1,depthwise3x3',
    [string]$MicroStrategies = 'direct,reuse',
    [string]$Blocks = 'audio_h2d,hubert,hubert_readback,rmvpe,retrieval_pitch,encoder,latent,flow,nsf_prepare,nsf,nsf_readback,rmvpe_unet,rmvpe_gru,rmvpe_head,nsf_upsample,nsf_noise_conv,nsf_residuals',
    [string]$PipelineStrategies = 'direct',
    [string]$Families = 'all',
    [switch]$NoTorch,
    [switch]$IncludeTorchFull,
    [switch]$IncludeNsys,
    [switch]$DryRun,
    [ValidateRange(1,8)][int]$NcuLaunchCount = 3,
    [ValidateRange(0,10000)][int]$NcuLaunchSkip = 0,
    [ValidateRange(2,64)][int]$MicroRuns = 5,
    [ValidateRange(2,16)][int]$PipelineRuns = 2,
    [ValidateRange(0,90)][int]$ProfileSeconds = 12,
    [ValidateRange(1,916)][int]$CoreFrames = 300,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [ValidateRange(0,1)][double]$IndexRate = 0.75,
    [ValidateRange(1,80)][int]$MaxTargets = 60,
    [string]$RustProbe = '.\stage103-conv2d-probe.exe',
    [string]$RustExe = '.\tts-with-rvc-rs-stage104.exe',
    [string]$PythonExe = 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe',
    [string]$OriginalRoot = '',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [string]$NcuExe = 'ncu',
    [string]$NsysExe = 'nsys'
)
$ErrorActionPreference = 'Stop'
$base = $PSScriptRoot
$runId = (Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss_ffff')
$reports = Join-Path $base "stage104_runs\$runId\reports"
$scratch = Join-Path $base "stage104_runs\$runId\scratch"
New-Item -ItemType Directory -Force -Path $reports,$scratch | Out-Null
$planScript = Join-Path $base 'stage104_matrix.py'
$trimScript = Join-Path $base 'trim_stage104_wav.py'
$torchMicro = Join-Path $base 'torch_stage104_conv2d.py'
$torchFull = Join-Path $base 'torch_stage102_profile.py'
$compare = Join-Path $base 'compare_stage103.py'
$planFile = Join-Path $reports 'capture-plan.json'
$manifestFile = Join-Path $reports 'manifest.json'
$oldEnv = @{}
$envNames = @('RVC_NVTX','RVC_STAGE103_STRATEGY','RVC_RMVPE_UNET_SPATIAL_REUSE',
    'RVC_RMVPE_UNET_SPATIAL_QUAD','RVC_RMVPE_UNET_GEMM',
    'RVC_RMVPE_UNET_CONV2D_GEMM','RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32',
    'RVC_RMVPE_UNET_COMPENSATED_TF32','RVC_RMVPE_UNET_WINOGRAD','RVC_RMVPE_BN_FOLD',
    'RVC_GLOBAL_F0','RVC_NSF_GLOBAL_PHASE','RVC_NSF_GLOBAL_NOISE',
    'RVC_CHUNK_STRATEGY','RVC_BATCH_SCHEDULE','PATH')
foreach ($name in $envNames) { $oldEnv[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
$success = $false
$statuses = New-Object System.Collections.Generic.List[object]
$inv = [Globalization.CultureInfo]::InvariantCulture

function Invoke-Logged([string]$Command, [string[]]$Arguments, [string]$Log) {
    Write-Host "Running $(Split-Path $Command -Leaf): $($Arguments -join ' ')"
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command @Arguments *>&1 | Out-File -LiteralPath $Log -Encoding UTF8
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($exitCode -ne 0) {
        Get-Content -LiteralPath $Log -Tail 40 | ForEach-Object { Write-Host $_ }
        throw "Process failed (exit $exitCode): $Command; log=$Log"
    }
}

function Set-PipelineMode([string]$Strategy) {
    foreach ($key in @('RVC_RMVPE_UNET_SPATIAL_REUSE','RVC_RMVPE_UNET_SPATIAL_QUAD',
             'RVC_RMVPE_UNET_GEMM','RVC_RMVPE_UNET_CONV2D_GEMM',
             'RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32','RVC_RMVPE_UNET_COMPENSATED_TF32',
             'RVC_RMVPE_UNET_WINOGRAD')) {
        [Environment]::SetEnvironmentVariable($key,$null,'Process')
    }
    $flags = @{
        reuse = 'RVC_RMVPE_UNET_SPATIAL_REUSE'
        quad = 'RVC_RMVPE_UNET_SPATIAL_QUAD'
        gemm = 'RVC_RMVPE_UNET_GEMM'
        tf32_batched = 'RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32'
        winograd = 'RVC_RMVPE_UNET_WINOGRAD'
    }
    if ($Strategy -ne 'direct') {
        if (-not $flags.ContainsKey($Strategy)) { throw "Unknown pipeline strategy: $Strategy" }
        [Environment]::SetEnvironmentVariable($flags[$Strategy],'1','Process')
    }
    $env:RVC_RMVPE_BN_FOLD = '1'
    $env:RVC_GLOBAL_F0 = '1'
    $env:RVC_NSF_GLOBAL_PHASE = '1'
    $env:RVC_NSF_GLOBAL_NOISE = '1'
    $env:RVC_CHUNK_STRATEGY = 'cost-aware'
    $env:RVC_BATCH_SCHEDULE = 'contiguous'
}

try {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
    $python = (Resolve-Path -LiteralPath $PythonExe).Path
    $planArgs = @($planScript,'--scope',$Scope,'--cases',$Cases,'--micro-strategies',$MicroStrategies,
        '--blocks',$Blocks,'--pipeline-strategies',$PipelineStrategies,'--families',$Families,
        '--max-targets',"$MaxTargets",'--output',$planFile)
    if ($NoTorch) { $planArgs += '--no-torch' }
    if ($IncludeTorchFull) { $planArgs += '--include-torch-full' }
    Invoke-Logged $python $planArgs (Join-Path $reports 'plan.log')
    $plan = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
    Write-Host "Stage104 NCU matrix: $($plan.target_count) captures, scope=$Scope, metrics=$MetricSet"
    if ($DryRun) {
        foreach ($t in $plan.targets) { Write-Host "  $($t.id): $($t.nvtx)" }
        return
    }
    if (-not (Get-Command $NcuExe -ErrorAction SilentlyContinue)) { throw "ncu not found: $NcuExe" }
    if ($IncludeNsys -and -not (Get-Command $NsysExe -ErrorAction SilentlyContinue)) { throw "nsys not found: $NsysExe" }
    $ncuVersion = & $NcuExe --version
    $ncuVersion | Out-File (Join-Path $reports 'ncu-version.txt') -Encoding utf8
    if ($LASTEXITCODE -ne 0) { throw 'ncu --version failed' }
    if ($IncludeNsys) { & $NsysExe --version | Out-File (Join-Path $reports 'nsys-version.txt') -Encoding utf8 }
    $venv = Split-Path (Split-Path $python -Parent) -Parent
    $nvtx = Join-Path $venv 'Lib\site-packages\torch\lib\nvToolsExt64_1.dll'
    if (Test-Path -LiteralPath $nvtx) { $env:PATH = "$(Split-Path $nvtx -Parent);$env:PATH" }
    $env:RVC_NVTX = '1'
    $microEnabled = $Scope -in @('micro','all')
    $fullEnabled = $Scope -in @('pipeline','all')
    if ($microEnabled) {
        if (-not (Test-Path -LiteralPath $RustProbe -PathType Leaf)) { throw "Missing probe: $RustProbe" }
        $rustProbePath = (Resolve-Path -LiteralPath $RustProbe).Path
        $probeNvtx = & $rustProbePath nvtx-status
        if ($LASTEXITCODE -ne 0 -or $probeNvtx -ne 'nvtx_available=true') { throw 'Stage104 Rust probe NVTX unavailable (nvToolsExt64_1.dll)' }
        foreach ($strategy in @('direct','reuse')) {
            $env:RVC_STAGE103_STRATEGY = $strategy
            Invoke-Logged $rustProbePath @('cuda',"$MicroRuns") (Join-Path $reports "micro-$strategy.json")
        }
        Invoke-Logged $python @($compare,'--direct',(Join-Path $reports 'micro-direct.json'),
            '--reuse',(Join-Path $reports 'micro-reuse.json'),'--output',(Join-Path $reports 'micro-parity.json')) (Join-Path $reports 'micro-parity.log')
        if (-not $NoTorch) {
            Invoke-Logged $python @($torchMicro,'--backend','cuda','--dtype','fp32',
                '--layout','nchw','--repeats',"$MicroRuns",'--output',(Join-Path $reports 'micro-torch.json')) (Join-Path $reports 'micro-torch.log')
        }
    }
    if ($fullEnabled) {
        foreach ($path in @($RustExe,$Voice,$Hubert,$Rmvpe,$InputWav)) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing full RVC input: $path" }
        }
        if ($IndexRate -gt 0 -and -not (Test-Path -LiteralPath $Index -PathType Leaf)) { throw "Missing index: $Index" }
        if ($CoreFrames + 84 -gt 1000) { throw 'CoreFrames + 84 must not exceed 1000' }
        $rustFullPath = (Resolve-Path -LiteralPath $RustExe).Path
        $fullNvtx = & $rustFullPath nvtx-status
        if ($LASTEXITCODE -ne 0 -or $fullNvtx -ne 'nvtx_available=true') { throw 'Stage104 full RVC NVTX unavailable' }
        $profileWav = (Resolve-Path -LiteralPath $InputWav).Path
        if ($ProfileSeconds -gt 0) {
            $profileWav = Join-Path $scratch 'profile-input-temporary.wav'
            Invoke-Logged $python @($trimScript,'--source',(Resolve-Path -LiteralPath $InputWav).Path,
                '--output',$profileWav,'--seconds',$ProfileSeconds.ToString($inv)) (Join-Path $reports 'trim.log')
        }
        $voicePath = (Resolve-Path -LiteralPath $Voice).Path
        $hubertPath = (Resolve-Path -LiteralPath $Hubert).Path
        $rmvpePath = (Resolve-Path -LiteralPath $Rmvpe).Path
        $indexPath = if ($IndexRate -eq 0) { '-' } else { (Resolve-Path -LiteralPath $Index).Path }
        if ($IncludeTorchFull) {
            if ($ProfileSeconds -gt 0) { throw 'PyTorch full pipeline requires -ProfileSeconds 0 to use exactly the same complete WAV' }
            if (-not $OriginalRoot -or -not (Test-Path (Join-Path $OriginalRoot 'tts_with_rvc/infer/vc/pipeline.py'))) {
                throw 'PyTorch full pipeline requires -OriginalRoot (unpacked upstream tts-with-rvc)'
            }
            $rootAssets = (Resolve-Path -LiteralPath (Split-Path $Voice -Parent)).Path
            foreach ($file in @($Hubert,$Rmvpe,$InputWav) + $(if ($IndexRate -gt 0) {@($Index)} else {@()})) {
                if ((Resolve-Path -LiteralPath (Split-Path $file -Parent)).Path -ne $rootAssets) {
                    throw 'PyTorch full pipeline requires all asset files including WAV in the same directory as Voice'
                }
            }
        }
        $sourceHash = (Get-FileHash -LiteralPath $InputWav -Algorithm SHA256).Hash.ToLowerInvariant()
        [pscustomobject]@{source_wav_sha256=$sourceHash; source_wav_name=(Split-Path $InputWav -Leaf);
            profile_seconds=$ProfileSeconds; index_rate=$IndexRate} | ConvertTo-Json | Set-Content (Join-Path $reports 'inputs.json') -Encoding UTF8
    }
    $sectionArgs = if ($MetricSet -eq 'full') { @('--set','full') } else {
        @('--section','SpeedOfLight','--section','LaunchStats','--section','Occupancy',
          '--section','MemoryWorkloadAnalysis','--section','ComputeWorkloadAnalysis',
          '--section','WarpStateStats','--section','SchedulerStats')
    }
    $nsysSeen = @{}
    $done = 0
    foreach ($target in $plan.targets) {
        $done += 1
        Write-Host "`n[Stage104 $done/$($plan.target_count)] $($target.id)"
        $id = $target.id
        if ($target.kind -eq 'micro_rust') {
            $env:RVC_STAGE103_STRATEGY = $target.strategy
            $command = $rustProbePath
            $commandArgs = @('cuda',"$MicroRuns",$target.case)
        } elseif ($target.kind -eq 'micro_torch') {
            $command = $python
            $commandArgs = @($torchMicro,'--backend','cuda','--dtype','fp32','--layout','nchw',
                '--repeats',"$MicroRuns",'--case',$target.case)
        } elseif ($target.kind -eq 'pipeline_rust') {
            Set-PipelineMode $target.strategy
            $command = $rustFullPath
            $wavOutput = Join-Path $scratch 'profile-output-temporary.wav'
            $commandArgs = @('infer-wav-full-benchmark',$voicePath,$hubertPath,$rmvpePath,$indexPath,
                $profileWav,$wavOutput,$IndexRate.ToString($inv),'cuda',"$PipelineRuns","$CoreFrames","$TensorBatchSize")
        } else {
            $command = $python
            $torchOutput = Join-Path $scratch 'torch-stage104-scratch'
            $commandArgs = @($torchFull,'--original-root',(Resolve-Path -LiteralPath $OriginalRoot).Path,
                '--assets-dir',$rootAssets,'--voice',(Split-Path $Voice -Leaf),
                '--hubert',(Split-Path $Hubert -Leaf),'--rmvpe',(Split-Path $Rmvpe -Leaf),
                '--index',(Split-Path $Index -Leaf),'--input-wav',(Split-Path $InputWav -Leaf),
                '--index-rate',$IndexRate.ToString($inv),'--precision','fp32',
                '--runs',"$PipelineRuns",'--output',$torchOutput)
        }
        if ($IncludeNsys) {
            $nsysKey = if ($target.kind -eq 'micro_rust' -or $target.kind -eq 'pipeline_rust') {
                "$($target.kind)-$($target.strategy)"
            } else { $target.kind }
            if (-not $nsysSeen.ContainsKey($nsysKey)) {
                $nsysSeen[$nsysKey] = $true
                $nsysOutput = Join-Path $reports "nsys-$id"
                Invoke-Logged $NsysExe (@('profile','--trace=cuda,nvtx','--sample=none',
                    '--force-overwrite=true','--output',$nsysOutput,$command) + $commandArgs) `
                    (Join-Path $reports "nsys-$id.log")
                if ($target.kind -eq 'pipeline_rust') {
                    Remove-Item -LiteralPath $wavOutput,"$wavOutput.benchmark.json" -Force -ErrorAction SilentlyContinue
                }
            }
        }
        $ncuOut = Join-Path $reports $id
        $log = Join-Path $reports "$id.log"
        $nvtxInclude = if ($target.kind -eq 'pipeline_rust') {
            "stage102_rust_measured/*/$($target.nvtx.TrimEnd('/'))"
        } else { $target.nvtx }
        $ncuArgs = @('--target-processes','all','--nvtx',
            '--nvtx-push-pop-scope','process','--nvtx-include',$nvtxInclude,
            '--launch-skip',"$NcuLaunchSkip",'--launch-count',"$NcuLaunchCount") +
            $sectionArgs + @('--export',$ncuOut,'--force-overwrite')
        if ($null -ne $target.kernel_filter -and $target.kernel_filter -ne '') {
            $ncuArgs += @('--kernel-name',$target.kernel_filter)
        }
        $ncuArgs += @($command) + $commandArgs
        Invoke-Logged $NcuExe $ncuArgs $log
        $rep = "$ncuOut.ncu-rep"
        $logBody = Get-Content -LiteralPath $log -Raw
        if ($logBody -match 'ERR_NVGPUCTRPERM') {
            throw 'NCU performance counters denied; run elevated PowerShell or grant counter permissions'
        }
        $profiled = (Test-Path -LiteralPath $rep -PathType Leaf) -and ($logBody -match '==PROF== Profiling')
        if ($profiled) {
            $state = 'profiled'
        } elseif ($logBody -match 'No kernels were profiled') {
            if ($target.kind -eq 'micro_rust' -or $target.kind -eq 'micro_torch') {
                throw "Microkernel capture $id contains zero kernels; check NVTX / process scope"
            }
            $state = 'no_kernels'
        } else {
            throw "NCU did not produce a valid report for $id; inspect $log"
        }
        $reportName = if ($profiled) { "$id.ncu-rep" } else { $null }
        $statuses.Add([pscustomobject]@{id=$id;state=$state;nvtx=$nvtxInclude;
            family=$target.family;kernel_filter=$target.kernel_filter;report=$reportName})
        Write-Host "NCU $id : $state"
        if ($target.kind -eq 'pipeline_rust') {
            $benchmark = "$wavOutput.benchmark.json"
            if (Test-Path -LiteralPath $benchmark) {
                Copy-Item -LiteralPath $benchmark -Destination (Join-Path $reports "$id-benchmark.json") -Force
            }
            Remove-Item -LiteralPath $wavOutput,$benchmark -Force -ErrorAction SilentlyContinue
        }
    }
    $captured = @($statuses | Where-Object { $_.state -eq 'profiled' }).Count
    if ($captured -eq 0) { throw 'NCU captured zero kernels across the entire matrix; result is not valid' }
    $success = $true
} finally {
    foreach ($name in $oldEnv.Keys) { [Environment]::SetEnvironmentVariable($name,$oldEnv[$name],'Process') }
    if (-not $DryRun) {
        [pscustomobject]@{stage=104; complete=$success; scope=$Scope; metric_set=$MetricSet;
            captured_utc=(Get-Date).ToUniversalTime().ToString('o');
            ncu_launch_count=$NcuLaunchCount; ncu_launch_skip=$NcuLaunchSkip;
            profiles=$statuses; output_audio_in_zip=$false;
            caveats=@('NCU replay can distort timing; compare unprofiled medians separately',
                'Only first selected launches in each NVTX block are profiled',
                'Torch and Rust full pipeline execution are not necessarily equivalent');
        } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestFile -Encoding UTF8
        if ($success) {
            $archive = Join-Path $base 'RVC_STAGE104_NCU_MATRIX_RESULTS.zip'
            if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
            $forbidden = Get-ChildItem -LiteralPath $reports -File -Recurse | Where-Object {
                $_.Extension -in @('.wav','.flac','.pth','.pt','.onnx','.index')
            }
            if ($forbidden) { throw 'Refusing to package audio, weights, or index into diagnostics ZIP' }
            Compress-Archive -Path (Join-Path $reports '*') -DestinationPath $archive -CompressionLevel Optimal
            Write-Host "Stage104 complete: $archive (no WAV/models/index)"
        } else {
            Write-Warning "Stage104 incomplete. Raw reports kept at $reports; no success ZIP created."
        }
    }
    Get-ChildItem -Path (Join-Path $scratch '*.wav') -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
}