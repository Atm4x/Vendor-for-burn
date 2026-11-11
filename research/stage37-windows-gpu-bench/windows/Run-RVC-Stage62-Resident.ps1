param(
    [ValidateSet('cpu','cuda','wgpu')][string]$Device = 'cuda',
    [string]$Voice = '.\embed\CrazyMita.pth',
    [string]$Hubert = '.\embed\hubert_base.pt',
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$Index = '.\embed\CrazyMita.index',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [double]$Start = 0.0,
    [double]$Duration = 0.7,
    [double]$IndexRate = 0.75,
    [ValidateRange(2,64)][int]$WarmRuns = 8
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'tts-with-rvc-rs.exe'
foreach ($p in @($exe,$Voice,$Hubert,$Rmvpe,$Index,$InputWav)) {
    if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing input: $p" }
}
if (-not ($Duration -ge 0.2 -and $Duration -le 9 -and $IndexRate -gt 0 -and $IndexRate -le 1)) {
    throw 'Stage62 requires 0.2..9 sec duration and 0<index_rate<=1'
}
$root = Join-Path $PSScriptRoot 'stage62_results'
$zip = Join-Path $PSScriptRoot 'RVC_STAGE62_RESIDENT_RESULTS.zip'
if (Test-Path $root) { Remove-Item $root -Recurse -Force }
New-Item -ItemType Directory -Path $root -Force | Out-Null
$vars = @('RVC_GPU_FAISS','RVC_GPU_F0','RVC_PROFILE')
$previous = @{}
foreach ($name in $vars) { $previous[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
$culture = [Globalization.CultureInfo]::InvariantCulture
$rows = @()
try {
    $cases = @(
        @{ name='cpu_index'; gpuFaiss=$false; gpuF0=$false },
        @{ name='resident_ivf'; gpuFaiss=$true; gpuF0=$false },
        @{ name='resident_ivf_f0'; gpuFaiss=$true; gpuF0=$true }
    )
    foreach ($case in $cases) {
        $env:RVC_GPU_FAISS = if ($case.gpuFaiss) { '1' } else { '0' }
        $env:RVC_GPU_F0 = if ($case.gpuF0) { '1' } else { '0' }
        if (-not $case.gpuFaiss) { Remove-Item Env:RVC_GPU_FAISS -ErrorAction SilentlyContinue }
        if (-not $case.gpuF0) { Remove-Item Env:RVC_GPU_F0 -ErrorAction SilentlyContinue }
        Remove-Item Env:RVC_PROFILE -ErrorAction SilentlyContinue
        $id = $case.name
        $wav = Join-Path $root "${id}.wav"
        $stdout = Join-Path $root "${id}.stdout.log"
        $stderr = Join-Path $root "${id}.stderr.log"
        $argsArray = @('infer-wav-segment-benchmark',$Voice,$Hubert,$Rmvpe,$Index,$InputWav,$wav,
            $Start.ToString($culture),$Duration.ToString($culture),$IndexRate.ToString($culture),'0',$Device,[string]$WarmRuns)
        $quoted = @($argsArray | ForEach-Object { '"' + $_.Replace('"','\"') + '"' })
        Write-Host "Stage62 ${id}: $Device, $WarmRuns repeated inferences in one process"
        $p = Start-Process -FilePath $exe -ArgumentList $quoted -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -Wait -PassThru -NoNewWindow
        if ($p.ExitCode -ne 0) { Get-Content $stderr; throw "Stage62 ${id} failed ($($p.ExitCode))" }
        $bench = Get-Content "${wav}.benchmark.json" -Raw | ConvertFrom-Json
        if ($bench.inference_ms.Count -ne $WarmRuns -or $null -eq $bench.warm_median_ms) { throw "Invalid warmed benchmark report for $id" }
        $row = [pscustomobject]@{
            mode = $id
            backend = $Device
            load_ms = $bench.load_ms
            first_inference_ms = $bench.inference_ms[0]
            warm_median_ms = $bench.warm_median_ms
            measurements_ms = $bench.inference_ms
            sha256 = (Get-FileHash $wav -Algorithm SHA256).Hash
        }
        $rows += $row
        Write-Host ("  load={0:N2}ms first={1:N2}ms warm_median={2:N2}ms" -f $row.load_ms,$row.first_inference_ms,$row.warm_median_ms)
    }
    $data = @{}
    foreach ($row in $rows) {
        $path = Join-Path $root ($row.mode + '.wav')
        $bytes = [IO.File]::ReadAllBytes($path)
        $offset = 12
        $samples = $null
        while ($offset + 8 -le $bytes.Length) {
            $chunk = [Text.Encoding]::ASCII.GetString($bytes,$offset,4)
            $size = [BitConverter]::ToInt32($bytes,$offset+4)
            if ($size -lt 0 -or $offset + 8 + $size -gt $bytes.Length) { throw "Invalid WAV chunk: $path" }
            if ($chunk -eq 'data') {
                $samples = New-Object 'System.Int16[]' ($size / 2)
                [Buffer]::BlockCopy($bytes,$offset+8,$samples,0,$size)
            }
            $offset += 8 + $size + ($size % 2)
        }
        if ($null -eq $samples) { throw "Missing PCM16 data in $path" }
        $data[$row.mode] = $samples
    }
    $ref = $data['cpu_index']
    $parities = @()
    foreach ($name in @('resident_ivf','resident_ivf_f0')) {
        $other = $data[$name]
        if ($other.Count -ne $ref.Count) { throw "PCM sample count mismatch ($name)" }
        $maxDelta = 0
        $bad = 0
        $sumSquares = 0.0
        for ($i=0; $i -lt $ref.Count; $i++) {
            $delta = [int]$other[$i] - [int]$ref[$i]
            $maxDelta = [Math]::Max($maxDelta,[Math]::Abs($delta))
            if ($delta -ne 0) { $bad++ }
            $sumSquares += [double]$delta * [double]$delta
        }
        $parities += [pscustomobject]@{mode=$name; max_abs_pcm16=$maxDelta; unequal_samples=$bad; samples=$ref.Count; rmse_pcm16=[Math]::Sqrt($sumSquares / [Math]::Max(1,$ref.Count))}
        Write-Host ("  $name parity: max_abs={0}, unequal={1}/{2}, rmse={3:N6}" -f $maxDelta,$bad,$ref.Count,[Math]::Sqrt($sumSquares / [Math]::Max(1,$ref.Count)))
    }
    [pscustomobject]@{stage=62; backend=$Device; warm_runs=$WarmRuns; graph_replay=$false; cases=$rows; parity=$parities} |
        ConvertTo-Json -Depth 15 | Set-Content (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($name in $vars) { [Environment]::SetEnvironmentVariable($name,$previous[$name],'Process') }
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Stage62 results: $zip"
}
