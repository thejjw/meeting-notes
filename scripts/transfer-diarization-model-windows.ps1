param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Backup", "Restore")]
    [string]$Action,
    [string]$Archive,
    [string]$Config,
    [ValidateSet("Optimal", "Fastest", "None")]
    [string]$CompressionLevel = "Optimal",
    [switch]$Force
)
$ErrorActionPreference = "Stop"

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    throw "uv is required and must be available on PATH. See https://docs.astral.sh/uv/getting-started/installation/"
}
if ($Action -eq "Restore" -and -not $Archive) {
    throw "-Archive is required when -Action Restore is selected."
}

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Arguments = @(
    "run", "--project", $RepositoryRoot,
    "python", "-m", "meeting_notes.model_transfer",
    "diarization", $Action.ToLowerInvariant(),
    "--compression-level", $CompressionLevel
)
if ($Archive) { $Arguments += @("--archive", $Archive) }
if ($Config) { $Arguments += @("--config", $Config) }
if ($Force) { $Arguments += "--force" }

& uv @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Diarization model $($Action.ToLowerInvariant()) failed with exit code $LASTEXITCODE."
}
