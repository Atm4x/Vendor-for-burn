param(
    [string]$Voice = ".\embed\CrazyMita.pth",
    [string]$Hubert = ".\embed\hubert_base.pt",
    [string]$Rmvpe = ".\embed\rmvpe.pt",
    [string]$Index = ".\embed\CrazyMita.index",
    [string]$InputWav = ".\embed\MitaKind.wav",
    [double]$Start = 0.0,
    [double]$Duration = 0.7,
    [double]$IndexRate = 0.75,
    [ValidateSet("cuda", "wgpu")][string]$Device = "cuda",
    [ValidateRange(1, 8)][int]$Rounds = 2
)

$ErrorActionPreference = "Stop"
$exe = Join-Path $PSScriptRoot "tts-with-rvc-rs.exe"
foreach ($path in @($exe, $Voice, $Hubert, $Rmvpe, $Index, $InputWav)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing file: $path" }
}
if ($Duration -lt 0.2 -or $Duration -gt 9.0 -or $Start -lt 0 -or $IndexRate -le 0 -or $IndexRate -gt 1) {
    throw "Stage61 requires a valid 0.2-9 s segment and nonzero index_rate"
}

function Read-Pcm16Data([string]$Path) {
    $data = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $Path).Path)
    if ($data.Length -lt 44 -or [Text.Encoding]::ASCII.GetString($data, 0, 4) -ne "RIFF") {
        throw "Unsupported WAV: $Path"
    }
    $cursor = 12
    $pcm = $null
    $format = $null
    while ($cursor + 8 -le $data.Length) {
        $name = [Text.Encoding]::ASCII.GetString($data, $cursor, 4)
        $size = [BitConverter]::ToInt32($data, $cursor + 4)
        if ($size -lt 0 -or $cursor + 8 + $size -gt $data.Length) { throw "Invalid WAV chunk: $Path" }
        if ($name -eq "fmt ") {
            if ($size -lt 16) { throw "Short fmt WAV: $Path" }
            $format = @([BitConverter]::ToInt16($data, $cursor + 8), [BitConverter]::ToInt16($data, $cursor + 10), [BitConverter]::ToInt16($data, $cursor + 22))
        }
        if ($name -eq "data") {
            if ($size % 2 -ne 0) { throw "Not PCM16 sample-aligned: $Path" }
            $pcm = New-Object 'System.Int16[]' ($size / 2)
            [Buffer]::BlockCopy($data, $cursor + 8, $pcm, 0, $size)
        }
        $cursor += 8 + $size + ($size % 2)
    }
    if ($null -eq $format -or $null -eq $pcm -or $format[0] -ne 1 -or $format[2] -ne 16) {
        throw "Only PCM16 output WAV supported: $Path"
    }
    return ,$pcm
}

$results = Join-Path $PSScriptRoot "stage61_results"
$archive = Join-Path $PSScriptRoot "RVC_STAGE61_GPU_FAISS_RESULTS.zip"
if (Test-Path $results) { Remove-Item $results -Recurse -Force }
New-Item -ItemType Directory -Force $results | Out-Null
$old = @{}
foreach ($name in @("RVC_GPU_FAISS", "RVC_PROFILE", "RVC_CONV_WEIGHT_PREPACK", "RVC_RMVPE_COMPACT_READBACK", "RVC_RMVPE_FUSED_READBACK", "RVC_RMVPE_PLANE_READBACK")) {
    $old[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$invariant = [Globalization.CultureInfo]::InvariantCulture
$rows = @()
$allExact = $true
try {
    if ($Device -eq "cuda") {
        try { & nvidia-smi -q > (Join-Path $results "nvidia-smi.txt") 2>&1 }
        catch { $_ | Out-String | Set-Content (Join-Path $results "nvidia-smi-error.txt") }
    }
    $env:RVC_PROFILE = "1"
    $env:RVC_CONV_WEIGHT_PREPACK = "1"
    Remove-Item Env:RVC_RMVPE_COMPACT_READBACK,Env:RVC_RMVPE_FUSED_READBACK,Env:RVC_RMVPE_PLANE_READBACK -ErrorAction SilentlyContinue
    for ($round = 1; $round -le $Rounds; $round++) {
        $order = if ($round % 2 -eq 1) { @("cpu-index", "gpu-index") } else { @("gpu-index", "cpu-index") }
        $roundRows = @()
        foreach ($mode in $order) {
            Remove-Item Env:RVC_GPU_FAISS -ErrorAction SilentlyContinue
            if ($mode -eq "gpu-index") { $env:RVC_GPU_FAISS = "1" }
            $id = "round${round}_${mode}"
            $stdout = Join-Path $results "${id}.stdout.log"
            $stderr = Join-Path $results "${id}.stderr.log"
            $wav = Join-Path $results "${id}.wav"
            $arguments = @("infer-wav-segment", $Voice, $Hubert, $Rmvpe, $Index, $InputWav, $wav,
                $Start.ToString($invariant), $Duration.ToString($invariant),
                $IndexRate.ToString($invariant), "0", $Device)
            $quoted = @($arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' })
            Write-Host "Running Stage61 ${id} ($Device) ..."
            $timer = [Diagnostics.Stopwatch]::StartNew()
            $process = Start-Process -FilePath $exe -ArgumentList $quoted -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -Wait -NoNewWindow
            $timer.Stop()
            if ($process.ExitCode -ne 0) {
                Get-Content -LiteralPath $stderr -ErrorAction SilentlyContinue
                throw "Stage61 ${id} failed (exit $($process.ExitCode))"
            }
            if (-not (Test-Path $wav)) { throw "Missing WAV for ${id}" }
            $log = Get-Content $stderr -Raw
            $hostMatch = [regex]::Match($log, 'profile hubert B=\d+ [\d.]+s, host_content_bytes=(?<bytes>\d+)')
            $retrieval = [regex]::Match($log, 'profile retrieval B=\d+ (?<sec>[\d.]+)s')
            if (-not $hostMatch.Success -or -not $retrieval.Success) { throw "Missing stage profile for ${id}" }
            $hostBytes = [int64]$hostMatch.Groups['bytes'].Value
            if (($mode -eq "gpu-index" -and $hostBytes -ne 0) -or ($mode -eq "cpu-index" -and $hostBytes -eq 0)) {
                throw "HuBERT host readback invariant failed for ${id} ($hostBytes bytes)"
            }
            $entry = [pscustomobject]@{
                id = $id
                mode = $mode
                device = $Device
                host_hubert_bytes = $hostBytes
                retrieval_profile_ms = [double]::Parse($retrieval.Groups['sec'].Value, $invariant) * 1000
                cold_total_ms = $timer.Elapsed.TotalMilliseconds
                wav_sha256 = (Get-FileHash -LiteralPath $wav -Algorithm SHA256).Hash
            }
            $rows += $entry
            $roundRows += $entry
            Write-Host ("  {0}: HuBERT D2H={1} B, retrieval={2:N2} ms, cold={3:N2} ms" -f $mode, $hostBytes, $entry.retrieval_profile_ms, $entry.cold_total_ms)
        }
        $baseline = Read-Pcm16Data (Join-Path $results "round${round}_cpu-index.wav")
        $candidate = Read-Pcm16Data (Join-Path $results "round${round}_gpu-index.wav")
        if ($baseline.Count -ne $candidate.Count) { throw "PCM16 output lengths differ in round $round" }
        $maxDiff = 0
        $squareSum = 0.0
        $unequal = 0
        for ($i = 0; $i -lt $baseline.Count; $i++) {
            $delta = [int]$baseline[$i] - [int]$candidate[$i]
            $maxDiff = [Math]::Max($maxDiff, [Math]::Abs($delta))
            $squareSum += [double]$delta * [double]$delta
            if ($delta -ne 0) { $unequal++ }
        }
        $exact = $unequal -eq 0
        $allExact = $allExact -and $exact
        Write-Host ("  PCM16 equal={0}, max_abs={1}, rmse={2:N6}, unequal={3}/{4}" -f $exact, $maxDiff, [Math]::Sqrt($squareSum / [Math]::Max(1, $baseline.Count)), $unequal, $baseline.Count)
        [pscustomobject]@{
            round = $round
            exact_pcm16 = $exact
            samples = $baseline.Count
            unequal_samples = $unequal
            max_abs_pcm16 = $maxDiff
            rmse_pcm16 = [Math]::Sqrt($squareSum / [Math]::Max(1, $baseline.Count))
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $results "round${round}_parity.json") -Encoding UTF8
    }
    [pscustomobject]@{
        stage = 61
        experiment = "CubeCL-resident-IVF-nprobe1-vs-CPU-IVF"
        device = $Device
        index_rate = $IndexRate
        duration = $Duration
        rounds = $Rounds
        all_pcm16_exact = $allExact
        measurements = $rows
        graph_replay = $false
    } | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $results "summary.json") -Encoding UTF8
}
finally {
    foreach ($name in $old.Keys) { [Environment]::SetEnvironmentVariable($name, $old[$name], "Process") }
    if (Test-Path $archive) { Remove-Item $archive -Force }
    Compress-Archive -Path (Join-Path $results '*') -DestinationPath $archive -CompressionLevel Optimal
    Write-Host "Results archive: $archive"
}
