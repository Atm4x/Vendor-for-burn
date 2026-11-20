param(
    [Parameter(Mandatory=$true)][string]$Voice,
    [Parameter(Mandatory=$true)][string]$Hubert,
    [Parameter(Mandatory=$true)][string]$Rmvpe,
    [string]$Index = "-",
    [Parameter(Mandatory=$true)][string]$InputWav,
    [double]$Start = 0.0,
    [double]$Duration = 2.0,
    [double]$IndexRate = 0.75,
    [double]$Pitch = 0.0,
    [ValidateSet("cpu", "cuda", "wgpu")][string]$Device = "cuda",
    [ValidateSet("baseline", "prepacked", "packed", "quad", "quad_interior")][string]$Variant = "prepacked"
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "tts-with-rvc-rs.exe"
foreach ($file in @($exe, $Voice, $Hubert, $Rmvpe, $InputWav)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
if ($Index -ne "-" -and -not (Test-Path -LiteralPath $Index -PathType Leaf)) {
    throw "Missing index: $Index"
}
if ($Start -lt 0 -or $Duration -lt 0.2 -or $Duration -gt 9.0) { throw "Invalid Start/Duration" }
if ($IndexRate -lt 0 -or $IndexRate -gt 1) { throw "Invalid IndexRate" }
if ($Index -eq "-") { $IndexRate = 0.0 }
$folder = Join-Path $PSScriptRoot "stage54_results"
New-Item -ItemType Directory -Force -Path $folder | Out-Null
$wavOut = Join-Path $folder "converted_${Device}_${Variant}.wav"
$stderrFile = Join-Path $folder "profile_${Device}_${Variant}.stderr.log"
$stdoutFile = Join-Path $folder "profile_${Device}_${Variant}.stdout.log"
$priorPrepack = [Environment]::GetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", "Process")
$priorKernel = [Environment]::GetEnvironmentVariable("RVC_NSF_CONV_KERNEL", "Process")
$priorProfile = [Environment]::GetEnvironmentVariable("RVC_PROFILE", "Process")
$elapsedMs = $null
try { & nvidia-smi -q > (Join-Path $folder "nvidia-smi.txt") 2>&1 }
catch { $_ | Out-String | Set-Content -LiteralPath (Join-Path $folder "nvidia-smi-error.txt") }
try {
    Remove-Item Env:RVC_NSF_CONV_KERNEL -ErrorAction SilentlyContinue
    Remove-Item Env:RVC_CONV_WEIGHT_PREPACK -ErrorAction SilentlyContinue
    if ($Variant -eq "prepacked") { $env:RVC_CONV_WEIGHT_PREPACK = "1" }
    if ($Variant -in @("packed", "quad", "quad_interior")) { $env:RVC_NSF_CONV_KERNEL = $Variant }
    $env:RVC_PROFILE = "1"
    $args = @("infer-wav-segment", $Voice, $Hubert, $Rmvpe, $Index, $InputWav, $wavOut,
        [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $Start),
        [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $Duration),
        [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $IndexRate),
        [string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $Pitch), $Device)
    Write-Host "Running RVC ${Device}/${Variant} with real model weights and audio..."
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $quoted = @($args | ForEach-Object { '"' + $_ + '"' })
    $process = Start-Process -FilePath $exe -ArgumentList $quoted -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile -Wait -PassThru -NoNewWindow
    $stopwatch.Stop()
    $elapsedMs = $stopwatch.Elapsed.TotalMilliseconds
    if ($process.ExitCode -ne 0) {
        Get-Content -LiteralPath $stderrFile -ErrorAction SilentlyContinue
        throw "RVC inference failed: exit=$($process.ExitCode)"
    }
    if (-not (Test-Path -LiteralPath $wavOut -PathType Leaf)) { throw "RVC did not produce WAV" }
    $fileSize = (Get-Item -LiteralPath $wavOut).Length
    if ($fileSize -le 44) { throw "RVC output WAV is empty" }
    $result = [pscustomobject]@{
        stage = 54
        backend = $Device
        variant = $Variant
        start_seconds = $Start
        duration_seconds = $Duration
        index_enabled = ($Index -ne "-")
        index_rate = $IndexRate
        output_wav = $wavOut
        output_bytes = $fileSize
        cold_total_ms = $elapsedMs
        graph_replay = $false
        production_cuda_parity = "not_verified"
    }
    $result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $folder "stage54_results_${Device}_${Variant}.json") -Encoding UTF8
    Write-Host "Success: $wavOut ($fileSize bytes), cold process ${elapsedMs}ms"
    Get-Content -LiteralPath $stderrFile
} finally {
    [Environment]::SetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", $priorPrepack, "Process")
    [Environment]::SetEnvironmentVariable("RVC_NSF_CONV_KERNEL", $priorKernel, "Process")
    [Environment]::SetEnvironmentVariable("RVC_PROFILE", $priorProfile, "Process")
    $archive = Join-Path $PSScriptRoot "RVC_STAGE54_RESULTS.zip"
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    Compress-Archive -Path (Join-Path $folder '*') -DestinationPath $archive -CompressionLevel Optimal
    Write-Host "Results: $archive"
}
