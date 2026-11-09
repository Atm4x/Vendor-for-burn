param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][string]$Hubert,
    [Parameter(Mandatory = $true)][string]$Rmvpe,
    [Parameter(Mandatory = $true)][string]$InputWav,
    [Parameter(Mandatory = $true)][string]$OutputWav,
    [string]$Index = "",
    [int]$BatchSize = 2,
    [float]$IndexRate = 0.75
)
$exe = Join-Path $PSScriptRoot "tts-with-rvc-rs.exe"
$argsList = @("convert", "--model", $Model, "--hubert", $Hubert, "--rmvpe", $Rmvpe, "--input", $InputWav, "--output", $OutputWav, "--batch-size", "$BatchSize")
if ($Index -ne "") { $argsList += @("--index", $Index, "--index-rate", "$IndexRate") }
& $exe @argsList
if ($LASTEXITCODE -ne 0) { throw "RVC conversion failed with code $LASTEXITCODE" }
