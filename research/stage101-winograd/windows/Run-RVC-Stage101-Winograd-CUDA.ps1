param(
    [string]$ReferenceExe = '.\tts-with-rvc-rs-stage101.exe',
    [string]$CandidateExe = '.\tts-with-rvc-rs-stage101.exe',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(0.0,1.0)][double]$IndexRate = 0.75,
    [ValidateRange(2,12)][int]$Runs = 4,
    [ValidateRange(1,916)][int]$CoreFrames = 900,
    [ValidateRange(1,32)][int]$TensorBatchSize = 2,
    [double]$TorchFp16ReferenceMs = 0,
    [string]$PythonExe = 'D:\Projects\tts-with-rvc\.venv\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$root = Join-Path $PSScriptRoot 'stage101_long_cuda'
New-Item -ItemType Directory -Force -Path $root | Out-Null
$analyzer = Join-Path $PSScriptRoot 'Analyze-Stage101-Winograd-CUDA.py'
$required = @($ReferenceExe, $CandidateExe, $Voice, $Hubert, $Rmvpe, $InputWav, $PythonExe, $analyzer)
if ($IndexRate -gt 0) { $required += $Index }
foreach ($file in $required) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
if ($CoreFrames + 84 -gt 1000) { throw 'CoreFrames + 84 must not exceed 1000' }
$ref = (Resolve-Path -LiteralPath $ReferenceExe).Path
$candidate = (Resolve-Path -LiteralPath $CandidateExe).Path
$voicePath = (Resolve-Path -LiteralPath $Voice).Path
$hubertPath = (Resolve-Path -LiteralPath $Hubert).Path
$rmvpePath = (Resolve-Path -LiteralPath $Rmvpe).Path
$wavPath = (Resolve-Path -LiteralPath $InputWav).Path
$indexPath = if ($IndexRate -eq 0) { '-' } else { (Resolve-Path -LiteralPath $Index).Path }
$duration = & $PythonExe -c 'import wave,sys; w=wave.open(sys.argv[1]);print(w.getnframes()/w.getframerate())' $wavPath
if ($LASTEXITCODE -ne 0) { throw 'Failed to read input WAV duration' }
if ([double]::Parse($duration,[Globalization.CultureInfo]::InvariantCulture) -lt 60) {
    throw 'Stage101 long-WAV acceptance requires at least 60 seconds of input audio'
}
$inv = [Globalization.CultureInfo]::InvariantCulture
$baselineEnv = @{
    RVC_NSF_IMPLICIT_GEMM = '1'
    RVC_RMVPE_GRU_PERSISTENT = '1'
    RVC_GLOBAL_F0 = '1'
    RVC_NSF_GLOBAL_PHASE = '1'
    RVC_NSF_GLOBAL_NOISE = '1'
    RVC_COST_LAUNCH = '10000'
    RVC_COST_LINEAR = '10'
    RVC_COST_ATTENTION_MILLI = '100'
    RVC_COST_SPLICE = '100'
    RVC_BATCH_LOOKAHEAD = '64'
    RVC_CHUNK_STRATEGY = 'cost-aware'
    RVC_BATCH_SCHEDULE = 'contiguous'
}
$cleared = @('RVC_RMVPE_BN_FOLD','RVC_RMVPE_UNET_SPATIAL_QUAD','RVC_RMVPE_UNET_WINOGRAD','RVC_RMVPE_UNET_GEMM',
    'RVC_RMVPE_UNET_CONV2D_GEMM','RVC_RMVPE_UNET_TRANSPOSE2D_GEMM',
    'RVC_RMVPE_UNET_COMPENSATED_TF32','RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32',
    'RVC_NSF_RESBLOCK_FP16','RVC_GPU_FAISS','RVC_GPU_F0',
    'RVC_DIAGNOSTIC_TRACE','RVC_PROFILE','CUDA_LAUNCH_BLOCKING',
    'RVC_RMVPE_MEL_REUSE','RVC_MEASURE_BATCH_LATENCY')
$affected = @($baselineEnv.Keys) + $cleared
$previous = @{}
foreach ($key in $affected) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
$cases = @('reference_direct','candidate_direct','fused_direct','fused_spatial','fused_tf32x2_batched','fused_winograd',
           'fused_winograd','fused_tf32x2_batched','fused_spatial','fused_direct','candidate_direct','reference_direct')
try {
    foreach ($key in $cleared) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    foreach ($key in $baselineEnv.Keys) { [Environment]::SetEnvironmentVariable($key,$baselineEnv[$key],'Process') }
    $fingerprints = [ordered]@{}
    foreach ($pair in @(@('reference_exe',$ref),@('candidate_exe',$candidate),@('voice',$voicePath),
         @('hubert',$hubertPath),@('rmvpe',$rmvpePath),@('input_wav',$wavPath))) {
        $fingerprints[$pair[0]] = (Get-FileHash -LiteralPath $pair[1] -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    if ($indexPath -ne '-') { $fingerprints['index'] = (Get-FileHash -LiteralPath $indexPath -Algorithm SHA256).Hash.ToLowerInvariant() }
    $torchMs = if ($TorchFp16ReferenceMs -gt 0) { $TorchFp16ReferenceMs } else { $null }
    $manifest = [ordered]@{
        stage=101; experiment='long-wav-cubecl-stage101-winograd-batched-gemm'
        input_duration_seconds=[double]::Parse($duration,[Globalization.CultureInfo]::InvariantCulture)
        runs_per_process=$Runs; core_frames=$CoreFrames
        tensor_batch_size=$TensorBatchSize; index_rate=$IndexRate
        torch_fp16_reference_ms=$torchMs
        order=$cases; rust_env=$baselineEnv; fingerprints=$fingerprints
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root 'manifest.json') -Encoding UTF8
    for ($i = 0; $i -lt $cases.Length; $i++) {
        $mode = $cases[$i]
        foreach ($key in $cleared) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
        $exe = if ($mode -eq 'reference_direct') { $ref } else { $candidate }
        if ($mode -like 'fused_*') {
            [Environment]::SetEnvironmentVariable('RVC_RMVPE_BN_FOLD','1','Process')
        }
        switch ($mode) {
            'fused_winograd' { [Environment]::SetEnvironmentVariable('RVC_RMVPE_UNET_WINOGRAD','1','Process') }
            'fused_spatial' { [Environment]::SetEnvironmentVariable('RVC_RMVPE_UNET_SPATIAL_QUAD','1','Process') }
            'fused_tf32x2_batched' { [Environment]::SetEnvironmentVariable('RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32','1','Process') }
        }
        $label = ('{0:00}_{1}' -f ($i+1), $mode)
        $output = Join-Path $root "$label.wav"
        $log = Join-Path $root "$label.log"
        $args = @('infer-wav-full-benchmark', $voicePath, $hubertPath, $rmvpePath,
            $indexPath, $wavPath, $output, $IndexRate.ToString($inv), 'cuda',
            "$Runs", "$CoreFrames", "$TensorBatchSize")
        Write-Host "Stage101 [$($i+1)/$($cases.Count)] $mode; runs=$Runs"
        $oldPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $exe @args *>&1 | Out-File -LiteralPath $log -Encoding UTF8
            $code = $LASTEXITCODE
        } finally { $ErrorActionPreference = $oldPreference }
        if ($code -ne 0) {
            Get-Content -LiteralPath $log -Tail 60 | ForEach-Object { Write-Host $_ }
            throw "Stage101 mode $mode failed code $code. Log: $log"
        }
        if (-not (Test-Path -LiteralPath "$output.benchmark.json")) {
            throw "No benchmark JSON: $label"
        }
    }
    & $PythonExe $analyzer --input $root --output (Join-Path $root 'analysis')
    if ($LASTEXITCODE -ne 0) { throw 'Stage101 analyzer failed' }
    $zip = Join-Path $PSScriptRoot 'RVC_STAGE101_WINOGRAD_CUDA_RESULTS.zip'
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Stage101 results: $zip"
} finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
}
