[CmdletBinding()]
param(
    [ValidateSet('cuda', 'cpu', 'wgpu', 'flex')]
    [string]$Device = 'cuda',
    [switch]$SkipGpu,
    [string]$RealModelPath,
    [string]$RealAudioPath,
    [string]$E2ECtcModelPath,
    [string]$E2ERnntModelPath,
    [string]$E2EAudioPath,
    [string]$ResultsDir,
    [ValidateRange(1, 86400)]
    [int]$TestTimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'gigaam-rs.exe'
$fixtureDir = Join-Path $PSScriptRoot 'fixtures/backend'
$audio = Join-Path $fixtureDir 'tone.wav'
if (-not $ResultsDir) { $ResultsDir = Join-Path $PSScriptRoot 'results' }
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing executable: $exe" }
if (-not (Test-Path -LiteralPath $audio -PathType Leaf)) { throw "Missing test WAV: $audio" }

foreach ($pathVariable in @('RealModelPath', 'RealAudioPath', 'E2ECtcModelPath', 'E2ERnntModelPath', 'E2EAudioPath')) {
    $value = Get-Variable -Name $pathVariable -ValueOnly
    if ($value) {
        $absolute = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($value)
        if (-not (Test-Path -LiteralPath $absolute -PathType Leaf)) {
            throw "File not found for -${pathVariable}: $absolute"
        }
        Set-Variable -Name $pathVariable -Value $absolute
    }
}

$results = [System.Collections.Generic.List[object]]::new()
$failed = 0

function ConvertTo-WindowsArgument {
    param([string]$Value)
    $builder = [System.Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Value.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * ($backslashes * 2 + 1)))
            [void]$builder.Append('"')
        } else {
            [void]$builder.Append(('\' * $backslashes))
            [void]$builder.Append($character)
        }
        $backslashes = 0
    }
    [void]$builder.Append(('\' * ($backslashes * 2)))
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Invoke-TestCase {
    param(
        [string]$Name,
        [string[]]$CommandArgs,
        [string]$RequiredFile,
        [switch]$RequireTranscription
    )
    $log = Join-Path $ResultsDir "$Name.log"
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $code = -1
    $stdout = ''
    $stderr = ''
    $process = $null
    try {
        $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $exe
        $startInfo.WorkingDirectory = $PSScriptRoot
        $startInfo.Arguments = ($CommandArgs | ForEach-Object { ConvertTo-WindowsArgument -Value $_ }) -join ' '
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        $startInfo.StandardOutputEncoding = [System.Text.UTF8Encoding]::new($false)
        $startInfo.StandardErrorEncoding = [System.Text.UTF8Encoding]::new($false)
        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) { throw 'Native process did not start.' }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $timedOut = -not $process.WaitForExit($TestTimeoutSeconds * 1000)
        if ($timedOut) {
            $process.Kill()
            $process.WaitForExit()
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $code = $process.ExitCode
        if ($timedOut) { throw "Test exceeded $TestTimeoutSeconds seconds" }
    } catch {
        $stderr += "`nProcess launch or I/O failed: $_"
    } finally {
        if ($null -ne $process) { $process.Dispose() }
        $watch.Stop()
    }
    $output = $stdout
    if ($stderr.Length -gt 0) {
        if ($output.Length -gt 0 -and -not $output.EndsWith("`n")) { $output += "`r`n" }
        $output += $stderr
    }
    [System.IO.File]::WriteAllText($log, $output, [System.Text.UTF8Encoding]::new($false))
    $filePresent = -not $RequiredFile -or ((Test-Path -LiteralPath $RequiredFile -PathType Leaf) -and ((Get-Item -LiteralPath $RequiredFile).Length -gt 0))
    $transcriptionPresent = -not $RequireTranscription -or ($stdout -match '(?m)^\[\d+\]\s+\S')
    $passed = ($code -eq 0) -and $filePresent -and $transcriptionPresent
    if (-not $passed) { $script:failed += 1 }
    $results.Add([pscustomobject]@{
        name = $Name
        passed = $passed
        exit_code = $code
        elapsed_ms = $watch.ElapsedMilliseconds
        log = $log
        required_output_exists = [bool]$filePresent
        required_transcription_exists = [bool]$transcriptionPresent
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
    Invoke-TestCase -Name 'ctc-v3-mel' -CommandArgs @('transcribe', (Join-Path $fixtureDir 'ctc-v3-mel'), $audio, '--device', $Device)
    Invoke-TestCase -Name 'rnnt' -CommandArgs @('transcribe', (Join-Path $fixtureDir 'rnnt'), $audio, '--device', $Device)
    $embeddings = Join-Path $ResultsDir 'ssl_embeddings.f32'
    Invoke-TestCase -Name 'ssl' -CommandArgs @('embed', (Join-Path $fixtureDir 'ssl'), $audio, $embeddings, '--device', $Device) -RequiredFile $embeddings
    Invoke-TestCase -Name 'emo' -CommandArgs @('emotion', (Join-Path $fixtureDir 'emo'), $audio, '--device', $Device)
    $parity = Join-Path $ResultsDir 'parity'
    Invoke-TestCase -Name 'ctc-parity' -CommandArgs @('parity', (Join-Path $fixtureDir 'ctc'), $audio, $parity, '--device', $Device) -RequiredFile (Join-Path $parity 'logits.f32')
    if ([bool]$RealModelPath -ne [bool]$RealAudioPath) { throw 'Pass both -RealModelPath and -RealAudioPath, or neither.' }
    if ($RealModelPath) {
        Invoke-TestCase -Name 'real-model-config' -CommandArgs @('inspect-config', $RealModelPath)
        Invoke-TestCase -Name 'real-model-asr' -CommandArgs @('transcribe', $RealModelPath, $RealAudioPath, '--device', $Device, '--profile') -RequireTranscription
    }
    if ($E2ECtcModelPath -or $E2ERnntModelPath) {
        if (-not $E2EAudioPath) { $E2EAudioPath = $RealAudioPath }
        if (-not $E2EAudioPath) { throw 'Pass -E2EAudioPath, or use -RealAudioPath with -E2ECtcModelPath / -E2ERnntModelPath.' }
        if ($E2ECtcModelPath) {
            Invoke-TestCase -Name 'e2e-ctc-config' -CommandArgs @('inspect-config', $E2ECtcModelPath)
            Invoke-TestCase -Name 'e2e-ctc-asr' -CommandArgs @('transcribe', $E2ECtcModelPath, $E2EAudioPath, '--device', $Device, '--profile') -RequireTranscription
        }
        if ($E2ERnntModelPath) {
            Invoke-TestCase -Name 'e2e-rnnt-config' -CommandArgs @('inspect-config', $E2ERnntModelPath)
            Invoke-TestCase -Name 'e2e-rnnt-asr' -CommandArgs @('transcribe', $E2ERnntModelPath, $E2EAudioPath, '--device', $Device, '--profile') -RequireTranscription
        }
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
