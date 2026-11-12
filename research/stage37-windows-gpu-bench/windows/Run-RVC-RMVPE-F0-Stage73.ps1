param(
    [string]$Rmvpe = '.\embed\rmvpe.pt',
    [string]$InputWav = '.\embed\MitaKind.wav',
    [ValidateRange(0,3600)][double]$StartSeconds = 80,
    [ValidateRange(0.5,10)][double]$DurationSeconds = 4
)
$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'profile_rmvpe_f0_stage73.exe'
foreach ($file in @($exe,$Rmvpe,$InputWav)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing input: $file" }
}
$exe = (Resolve-Path -LiteralPath $exe).Path
$Rmvpe = (Resolve-Path -LiteralPath $Rmvpe).Path
$InputWav = (Resolve-Path -LiteralPath $InputWav).Path
$root = Join-Path $PSScriptRoot 'stage73_rmvpe_f0_results'
$zip = Join-Path $PSScriptRoot 'RVC_STAGE73_RMVPE_F0_RESULTS.zip'
$keys = @(
    'RVC_RMVPE_UNET_GEMM','RVC_RMVPE_UNET_CONV2D_GEMM',
    'RVC_RMVPE_UNET_TRANSPOSE2D_GEMM','RVC_RMVPE_GRU_PERSISTENT','RVC_DIAGNOSTIC_TRACE'
)
$previous = @{}
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key,'Process') }
if (Test-Path -LiteralPath $root) { Remove-Item -LiteralPath $root -Recurse -Force }
New-Item -ItemType Directory -Force -Path $root | Out-Null
$cases = @(
    [pscustomobject]@{name='direct';conv=$false;transpose=$false},
    [pscustomobject]@{name='conv2d';conv=$true;transpose=$false},
    [pscustomobject]@{name='transpose2d';conv=$false;transpose=$true},
    [pscustomobject]@{name='both';conv=$true;transpose=$true}
)
$rows = @()
$parity = @()
$failures = @()
$culture = [Globalization.CultureInfo]::InvariantCulture
try {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$null,'Process') }
    $env:RVC_RMVPE_GRU_PERSISTENT = '1'
    foreach ($case in $cases) {
        $name = $case.name
        try {
            Remove-Item Env:RVC_RMVPE_UNET_CONV2D_GEMM,Env:RVC_RMVPE_UNET_TRANSPOSE2D_GEMM -ErrorAction SilentlyContinue
            if ($case.conv) { $env:RVC_RMVPE_UNET_CONV2D_GEMM = '1' }
            if ($case.transpose) { $env:RVC_RMVPE_UNET_TRANSPOSE2D_GEMM = '1' }
            $prefix = Join-Path $root $name
            $stdout = Join-Path $root "$name.stdout.json"
            $stderr = Join-Path $root "$name.stderr.log"
            Write-Host "Stage73 real RMVPE: $name"
            & $exe $Rmvpe $InputWav $prefix $StartSeconds.ToString($culture) $DurationSeconds.ToString($culture) 1> $stdout 2> $stderr
            if ($LASTEXITCODE -ne 0) { throw "$name CUDA process failed with exit code $LASTEXITCODE" }
            $report = Get-Content -LiteralPath $stdout -Raw | ConvertFrom-Json
            if ($report.stage -ne 73 -or $report.gru_persistent -ne $true -or
                $report.unet_conv2d_gemm -ne $case.conv -or
                $report.unet_transpose2d_gemm -ne $case.transpose -or $report.frames -lt 50) {
                throw "$name: wrong execution metadata"
            }
            $rows += [pscustomobject]@{
                mode=$name; frames=$report.frames; first_ms=$report.first_salience_ms;
                warm_ms=$report.warm_salience_ms; voiced_frames=$report.voiced_frames;
                salience_sha256=(Get-FileHash -LiteralPath "$prefix.salience.f32le" -Algorithm SHA256).Hash;
                f0_sha256=(Get-FileHash -LiteralPath "$prefix.f0.f32le" -Algorithm SHA256).Hash
            }
        } catch {
            $failures += [pscustomobject]@{mode=$name;error=$_.Exception.Message}
            Write-Warning "Stage73 RMVPE $name failed: $($_.Exception.Message)"
        }
    }
    if (@($rows | Where-Object mode -eq 'direct').Count -eq 1) {
        foreach ($kind in @('salience','f0')) {
            $aBytes = [IO.File]::ReadAllBytes((Join-Path $root "direct.$kind.f32le"))
            foreach ($row in @($rows | Where-Object mode -ne 'direct')) {
                $bBytes = [IO.File]::ReadAllBytes((Join-Path $root "$($row.mode).$kind.f32le"))
                if ($aBytes.Length -ne $bBytes.Length -or ($aBytes.Length % 4) -ne 0) {
                    throw "$($row.mode) $kind: incompatible FP32 array lengths"
                }
                $count = [int]($aBytes.Length / 4)
                $a = [float[]]::new($count)
                $b = [float[]]::new($count)
                [Buffer]::BlockCopy($aBytes,0,$a,0,$aBytes.Length)
                [Buffer]::BlockCopy($bBytes,0,$b,0,$bBytes.Length)
                $changed = 0; $max = 0.0; $sum = 0.0; $voicing = 0
                for ($i=0;$i -lt $count;$i++) {
                    if ($a[$i] -ne $b[$i]) { $changed++ }
                    $delta = [double]$a[$i] - [double]$b[$i]
                    if ([double]::IsNaN($delta) -or [double]::IsInfinity($delta)) { throw "$kind nonfinite" }
                    $max = [Math]::Max($max,[Math]::Abs($delta))
                    $sum += $delta*$delta
                    if ($kind -eq 'f0' -and (($a[$i] -gt 0) -ne ($b[$i] -gt 0))) { $voicing++ }
                }
                $parity += [pscustomobject]@{
                    mode=$row.mode; signal=$kind; samples=$count;
                    unequal=$changed; rmse=[Math]::Sqrt($sum/$count);
                    max_abs=$max; voiced_unvoiced_changes=$voicing
                }
            }
        }
    }
    [pscustomobject]@{
        stage=73; start_seconds=$StartSeconds; duration_seconds=$DurationSeconds;
        experiment='Actual RMVPE salience + F0 before RVC synthesis'; graph_replay=$false;
        cases=$rows; parity=$parity; failures=$failures
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $root 'summary.json') -Encoding UTF8
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key,$previous[$key],'Process') }
    if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
    Compress-Archive -Path (Join-Path $root '*') -DestinationPath $zip -CompressionLevel Optimal
    Write-Host "Stage73 RMVPE F0 ZIP: $zip"
}
