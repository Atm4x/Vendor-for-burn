[CmdletBinding()]
param(
    [ValidateSet('transcribe', 'transcribe-batch', 'embed', 'emotion', 'parity', 'parity-encoder', 'probe-audio', 'inspect-checkpoint', 'inspect-config')]
    [string]$Mode = 'transcribe',
    [string]$ModelPath,
    [string]$AudioPath,
    [string]$BatchJson,
    [string]$OutputPath,
    [string]$SegmentsJson,
    [ValidateSet('cuda', 'cpu', 'wgpu', 'flex')]
    [string]$Device = 'cuda',
    [switch]$WordTimestamps,
    [switch]$GroupVadPython,
    [switch]$Raw,
    [switch]$All,
    [switch]$Profile
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot 'gigaam-rs.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Missing executable: $exe. Extract the complete GitHub Actions ZIP first."
}

$cliArgs = @()
switch ($Mode) {
    'probe-audio' {
        if (-not $AudioPath) { $AudioPath = Join-Path $PSScriptRoot 'fixtures/backend/tone.wav' }
        $cliArgs = @('probe-audio', $AudioPath)
    }
    'inspect-checkpoint' {
        if (-not $ModelPath) { throw '-ModelPath must point to a .ckpt or .safetensors file.' }
        $cliArgs = @('inspect-checkpoint', $ModelPath)
        if ($All) { $cliArgs += '--all' }
    }
    'inspect-config' {
        if (-not $ModelPath) { throw '-ModelPath must point to a .ckpt file.' }
        $cliArgs = @('inspect-config', $ModelPath)
        if ($Raw) { $cliArgs += '--raw' }
    }
    'transcribe-batch' {
        if (-not $ModelPath -or -not $BatchJson) { throw 'Both -ModelPath and -BatchJson are required.' }
        $cliArgs = @('transcribe-batch', $ModelPath, $BatchJson, '--device', $Device)
    }
    default {
        if (-not $ModelPath -or -not $AudioPath) { throw 'Both -ModelPath and -AudioPath are required.' }
        $cliArgs = @($Mode, $ModelPath, $AudioPath)
        switch ($Mode) {
            'transcribe' {
                if ($WordTimestamps) { $cliArgs += '--word-timestamps' }
                if ($SegmentsJson) { $cliArgs += @('--segments-json', $SegmentsJson) }
                if ($GroupVadPython) {
                    if (-not $SegmentsJson) { throw '-GroupVadPython requires -SegmentsJson.' }
                    $cliArgs += '--group-vad-python'
                }
            }
            'embed' {
                if (-not $OutputPath) { throw '-OutputPath is required for embed.' }
                $cliArgs += $OutputPath
            }
            'parity' {
                if (-not $OutputPath) { throw '-OutputPath is required for parity.' }
                $cliArgs += $OutputPath
            }
            'parity-encoder' {
                if (-not $OutputPath) { throw '-OutputPath is required for parity-encoder.' }
                $cliArgs += $OutputPath
            }
        }
        $cliArgs += @('--device', $Device)
        if ($Profile -and $Mode -eq 'transcribe') { $cliArgs += '--profile' }
    }
}

Write-Host "Command: $Mode | Backend: $Device | Executable: $exe"
$previousOutputEncoding = [Console]::OutputEncoding
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    & $exe @cliArgs
    $exitCode = $LASTEXITCODE
} finally {
    [Console]::OutputEncoding = $previousOutputEncoding
}
if ($exitCode -ne 0) {
    throw "gigaam-rs exited with code $exitCode (mode=$Mode)."
}
