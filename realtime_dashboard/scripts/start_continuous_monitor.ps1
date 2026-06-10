param(
    [string]$Universe = "full_market",
    [int]$Workers = 64,
    [string]$ScanMode = "chunked",
    [switch]$Enable5m = $true,
    [switch]$Enable15m = $true,
    [switch]$PublishRuntime = $true
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
Set-Location $repoRoot

$python = "python"
$args = @(
    "realtime_dashboard/scripts/continuous_monitor.py",
    "--universe", $Universe,
    "--workers", $Workers,
    "--scan-mode", $ScanMode
)

if ($Enable15m) {
    $args += "--enable-15m"
}

if ($Enable5m) {
    $args += "--enable-5m"
}

if ($PublishRuntime) {
    $args += "--github-push"
}

& $python $args
