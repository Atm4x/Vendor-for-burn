[CmdletBinding()]
param(
    [ValidateSet('cuda', 'cpu', 'wgpu', 'flex')]
    [string]$Device = 'cuda',
    [switch]$SkipGpu,
    [string]$RealModelPath,
    [string]$RealAudioPath,
    [string]$ResultsDir
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'gigaam-rs.exe'
$fixtureDir = Join-Path $PSScriptRoot 'fixtures/backend'
$audio = Join-Path $fixtureDir 'tone.wav'
if (-not $ResultsDir) { $ResultsDir = Join-Path $PSScriptRoot 'results' }
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing executable: $exe" }
if (-not (Test-Path -LiteralPath $audio -PathType Leaf)) { throw "Missing test WAV: $audio" }

$results = [System.Collections.Generic.List[object]]::new()
$failed = 0

function Invoke-TestCase {
    param(
        [string]$Name,
        [string[]]$CommandArgs,
        [string]$RequiredFile
    )
    $log = Join-Path $ResultsDir "$Name.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $output = (& $exe @CommandArgs 2>&1 | Out-String)
    $code = $LASTEXITCODE
    $watch.Stop()
    $filePresent = -not $RequiredFile -or ((Test-Path -LiteralPath $RequiredFile -PathType Leaf) -and ((Get-Item -LiteralPath $RequiredFile).Length -gt 0))
    $passed = ($code -eq 0) -and $filePresent
    $output | Set-Content -LiteralPath $log -Encoding utf8
    if (-not $passed) { $script:failed += 1 }
    $results.Add([pscustomobject]@{
        name = $Name
        passed = $passed
        exit_code = $code
        elapsed_ms = $watch.ElapsedMilliseconds
        log = $log
        required_output_exists = [bool]$filePresent
    })
    if ($passed) { Write-Host "[PASS] $Name ($($watch.ElapsedMilliseconds) ms)" -ForegroundColor Green }
    else { Write-Host "[FAIL] $Name (exit=$code) | $log" -ForegroundColor Red }
}

Invoke-TestCase -Name 'audio-probe' -CommandArgs @('probe-audio', $audio)
Invoke-TestCase -Name 'checkpoint-inspect' -CommandArgs @('inspect-checkpoint', (Join-Path $fixtureDir 'ctc/model.safetensors'))

if (-not $SkipGpu) {
    if ($Device -eq 'cuda') {
        $gpuInfo = (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
        if ($gpuInfo) {
            & nvidia-smi | Out-File -LiteralPath (Join-Path $ResultsDir 'nvidia-smi.txt') -Encoding utf8
        } else {
            Write-Warning 'nvidia-smi was not found. CUDA checks will still run and report their actual result.'
        }
    }
    Invoke-TestCase -Name 'ctc' -CommandArgs @('transcribe', (Join-Path $fixtureDir 'ctc'), $audio, '--device', $Device)
    Invoke-TestCase -Name 'rnnt' -CommandArgs @('transcribe', (Join-Path $fixtureDir 'rnnt'), $audio, '--device', $Device)
    $embeddings = Join-Path $ResultsDir 'ssl_embeddings.f32'
    Invoke-TestCase -Name 'ssl' -CommandArgs @('embed', (Join-Path $fixtureDir 'ssl'), $audio, $embeddings, '--device', $Device) -RequiredFile $embeddings
    Invoke-TestCase -Name 'emo' -CommandArgs @('emotion', (Join-Path $fixtureDir 'emo'), $audio, '--device', $Device)
    $parity = Join-Path $ResultsDir 'parity'
    Invoke-TestCase -Name 'ctc-parity' -CommandArgs @('parity', (Join-Path $fixtureDir 'ctc'), $audio, $parity, '--device', $Device) -RequiredFile (Join-Path $parity 'logits.f32')
    if ([bool]$RealModelPath -ne [bool]$RealAudioPath) { throw 'Pass both -RealModelPath and -RealAudioPath, or neither.' }
    if ($RealModelPath) {
        Invoke-TestCase -Name 'real-model-asr' -CommandArgs @('transcribe', $RealModelPath, $RealAudioPath, '--device', $Device)
    }
} else {
    Write-Host 'GPU inference skipped by request. Only CPU-independent audio/checkpoint inspection was tested.'
}

$summary = [pscustomobject]@{
    timestamp = (Get-Date).ToString('o')
    backend = $Device
    gpu_inference_skipped = [bool]$SkipGpu
    platform = 'windows-x64'
    passed = ($failed -eq 0)
    results = @($results.ToArray())
}
$summaryPath = Join-Path $ResultsDir 'summary.json'
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding utf8
Write-Host "Summary: $summaryPath"
if ($failed -gt 0) { throw "$failed of $($results.Count) GigaAM tests failed. Attach the results directory to your report." }
