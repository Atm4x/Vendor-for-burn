param(
    [Parameter(Mandatory=$true)][string]$Voice,
    [Parameter(Mandatory=$true)][string]$Hubert,
    [Parameter(Mandatory=$true)][string]$Rmvpe,
    [Parameter(Mandatory=$true)][string]$InputWav,
    [string]$Index = "-",
    [double]$Start = 0.0,
    [double]$Duration = 0.7,
    [double]$IndexRate = 0.75,
    [double]$Pitch = 0.0,
    [ValidateSet("cuda")][string]$Device = "cuda",
    [ValidateRange(1, 8)][int]$Rounds = 2
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "tts-with-rvc-rs.exe"
foreach ($file in @($exe, $Voice, $Hubert, $Rmvpe, $InputWav)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing file: $file" }
}
if ($Index -ne "-" -and -not (Test-Path -LiteralPath $Index -PathType Leaf)) {
    throw "Missing FAISS index: $Index"
}
if ($Start -lt 0 -or $Duration -lt 0.2 -or $Duration -gt 9.0) { throw "Invalid segment range" }
if ($IndexRate -lt 0 -or $IndexRate -gt 1) { throw "Invalid IndexRate" }
if ($Index -eq "-") { $IndexRate = 0.0 }

$folder = Join-Path $PSScriptRoot "stage59_results"
New-Item -ItemType Directory -Force -Path $folder | Out-Null
$archive = Join-Path $PSScriptRoot "RVC_STAGE59_RMVPE_FUSED_RESULTS.zip"
$priorCompact = [Environment]::GetEnvironmentVariable("RVC_RMVPE_COMPACT_READBACK", "Process")
$priorFused = [Environment]::GetEnvironmentVariable("RVC_RMVPE_FUSED_READBACK", "Process")
$priorPrepack = [Environment]::GetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", "Process")
$priorProfile = [Environment]::GetEnvironmentVariable("RVC_PROFILE", "Process")
$priorKernel = [Environment]::GetEnvironmentVariable("RVC_NSF_CONV_KERNEL", "Process")
$invariant = [System.Globalization.CultureInfo]::InvariantCulture
$rows = @()
try {
    try { & nvidia-smi -q > (Join-Path $folder "nvidia-smi.txt") 2>&1 }
    catch { $_ | Out-String | Set-Content -LiteralPath (Join-Path $folder "nvidia-smi-error.txt") }
    $env:RVC_PROFILE = "1"
    $env:RVC_CONV_WEIGHT_PREPACK = "1"
    Remove-Item Env:RVC_NSF_CONV_KERNEL -ErrorAction SilentlyContinue
    for ($round = 1; $round -le $Rounds; $round++) {
        $modes = if ($round % 2 -eq 1) { @("dense", "compact", "fused") } else { @("fused", "compact", "dense") }
        $roundRows = @()
        foreach ($mode in $modes) {
            Remove-Item Env:RVC_RMVPE_COMPACT_READBACK -ErrorAction SilentlyContinue
            Remove-Item Env:RVC_RMVPE_FUSED_READBACK -ErrorAction SilentlyContinue
            if ($mode -eq "compact") { $env:RVC_RMVPE_COMPACT_READBACK = "1" }
            if ($mode -eq "fused") { $env:RVC_RMVPE_FUSED_READBACK = "1" }
            $id = "round${round}_${mode}"
            $stdoutFile = Join-Path $folder "${id}.stdout.log"
            $stderrFile = Join-Path $folder "${id}.stderr.log"
            $output = Join-Path $folder "${id}.wav"
            $argsList = @("infer-wav-segment", $Voice, $Hubert, $Rmvpe, $Index, $InputWav, $output,
                $Start.ToString($invariant), $Duration.ToString($invariant),
                $IndexRate.ToString($invariant), $Pitch.ToString($invariant), $Device)
            $quoted = @($argsList | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' })
            Write-Host ("Running Stage59 {0} with {1} ..." -f $id, $Device)
            $watch = [System.Diagnostics.Stopwatch]::StartNew()
            $process = Start-Process -FilePath $exe -ArgumentList $quoted -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile -PassThru -Wait -NoNewWindow
            $watch.Stop()
            if ($process.ExitCode -ne 0) {
                Get-Content -LiteralPath $stderrFile -ErrorAction SilentlyContinue
                throw "Stage59 ${id} failed: exit $($process.ExitCode)"
            }
            if (-not (Test-Path -LiteralPath $output -PathType Leaf) -or (Get-Item -LiteralPath $output).Length -le 44) {
                throw "No valid output WAV for ${id}"
            }
            $log = Get-Content -LiteralPath $stderrFile -Raw
            $readback = [regex]::Match($log, 'profile rmvpe_f0_readback (?<mode>dense|compact|fused) B=(?<batch>\d+) frames=(?<frames>\d+) host_bytes=(?<bytes>\d+)')
            if (-not $readback.Success -or $readback.Groups['mode'].Value -ne $mode) { throw "Missing RMVPE readback evidence for ${id}" }
            $frames = [int]$readback.Groups['frames'].Value
            $batch = [int]$readback.Groups['batch'].Value
            $bytes = [int]$readback.Groups['bytes'].Value
            $expectedBytes = $frames * $batch * $(if ($mode -eq "dense") { 360 } else { 10 }) * 4
            if ($bytes -ne $expectedBytes) { throw "Unexpected readback bytes for ${id}: $bytes vs $expectedBytes" }
            $rmvpeMatch = [regex]::Match($log, 'profile rmvpe B=\d+ (?<seconds>[\d.]+)s')
            if (-not $rmvpeMatch.Success) { throw "Missing RMVPE profile for ${id}" }
            $entry = [pscustomobject]@{
                id = $id
                mode = $mode
                backend = $Device
                frames = $frames
                batch = $batch
                host_bytes = $bytes
                rmvpe_stage_ms = [double]::Parse($rmvpeMatch.Groups['seconds'].Value, $invariant) * 1000.0
                cold_total_ms = $watch.Elapsed.TotalMilliseconds
                wav_sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
            }
            $roundRows += $entry
            $rows += $entry
            Write-Host ("  {0}: RMVPE {1:N2}ms, D2H {2} bytes, WAV SHA256 {3}" -f $mode, $entry.rmvpe_stage_ms, $bytes, $entry.wav_sha256)
        }
        if (@($roundRows | Select-Object -ExpandProperty wav_sha256 -Unique).Count -ne 1) {
            throw "Stage59 FP32-to-WAV parity failed in round ${round}"
        }
    }
    $summary = [pscustomobject]@{
        stage = 59
        experiment = "three-way-fused-RMVPE-salience-readback"
        device = $Device
        rounds = $Rounds
        index_rate = $IndexRate
        parity_pass = $true
        input_segment_start = $Start
        input_segment_seconds = $Duration
        graph_replay = $false
        measurements = $rows
    }
    $summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $folder "summary.json") -Encoding UTF8
    Write-Host "Stage59: all rounds passed WAV parity"
}
finally {
    [Environment]::SetEnvironmentVariable("RVC_RMVPE_COMPACT_READBACK", $priorCompact, "Process")
    [Environment]::SetEnvironmentVariable("RVC_RMVPE_FUSED_READBACK", $priorFused, "Process")
    [Environment]::SetEnvironmentVariable("RVC_CONV_WEIGHT_PREPACK", $priorPrepack, "Process")
    [Environment]::SetEnvironmentVariable("RVC_PROFILE", $priorProfile, "Process")
    [Environment]::SetEnvironmentVariable("RVC_NSF_CONV_KERNEL", $priorKernel, "Process")
    if (Test-Path -LiteralPath $archive) { Remove-Item -LiteralPath $archive -Force }
    Compress-Archive -Path (Join-Path $folder '*') -DestinationPath $archive -CompressionLevel Optimal
    Write-Host "Results archive: $archive"
}
