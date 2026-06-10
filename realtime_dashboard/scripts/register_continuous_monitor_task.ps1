param(
    [string]$TaskName = "StockNetContinuousMonitor",
    [string]$Universe = "core_500",
    [int]$Workers = 64,
    [string]$ScanMode = "full_parallel"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$startScript = Join-Path $scriptDir "start_continuous_monitor.ps1"
$workingDir = Split-Path -Parent (Split-Path -Parent $scriptDir)

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$startScript`" -Universe $Universe -Workers $Workers -ScanMode $ScanMode"

$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Run the StockNet continuous market monitor in the background."
