[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][ValidatePattern('^LLMSimThread_[a-f0-9]{32}$')][string]$InstanceName,
    [ValidateRange(10,60)][int]$MaximumTraceSeconds=30,
    [switch]$Capture
)
$ErrorActionPreference='Stop'
$stagingRoot=[IO.Path]::GetFullPath($PSScriptRoot)
$captureRoot=[IO.Path]::GetFullPath($OutDir)
if (-not $captureRoot.StartsWith($stagingRoot+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)) { throw 'Output must be a fresh child directory of this staging directory.' }
if (Test-Path -LiteralPath $captureRoot) { throw 'Output already exists; never replace a previous trace.' }
[IO.Directory]::CreateDirectory($captureRoot) | Out-Null
$utf8=New-Object Text.UTF8Encoding($false)
function SaveJson([string]$Name,$Value) {
    $path=Join-Path $captureRoot $Name
    $file=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try {
        $writer=New-Object IO.StreamWriter($file,$utf8)
        try { $writer.Write(($Value | ConvertTo-Json -Depth 20).Replace("`r`n","`n")+"`n") } finally { $writer.Dispose() }
    } finally { $file.Dispose() }
}
function InvokeTool([string]$Label,[string]$Tool,[string[]]$Arguments) {
    $before=[Diagnostics.Stopwatch]::GetTimestamp()
    $output=(& $Tool @Arguments 2>&1 | Out-String)
    $exitCode=$LASTEXITCODE
    $after=[Diagnostics.Stopwatch]::GetTimestamp()
    $row=[ordered]@{tool=$Tool;arguments=$Arguments;exit_code=$exitCode;stdout_stderr=$output;qpc_before=$before;qpc_after=$after;qpc_frequency=[Diagnostics.Stopwatch]::Frequency}
    SaveJson ($Label+'.json') $row
    return $row
}
function NeedSuccess($Result) { if ($Result.exit_code -ne 0) { throw ('Tool failed: '+$Result.tool+' exit '+$Result.exit_code) } }
$wpr=(Get-Command wpr.exe -ErrorAction Stop).Source
$xperf=(Get-Command xperf.exe -ErrorAction Stop).Source
$profile=Join-Path $stagingRoot 'OwnedThreads.wprp'
$tracePath=Join-Path $captureRoot 'owned-thread-trace.etl'
$stopPath=Join-Path $captureRoot 'stop.request.json'
$isAdmin=(New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$privileges=InvokeTool 'token-privileges' (Get-Command whoami.exe).Source @('/priv')
$hasProfilePrivilege=$privileges.stdout_stderr -match 'SeSystemProfilePrivilege'
$inventory=InvokeTool 'global-loggers-before' $xperf @('-loggers')
NeedSuccess $inventory
$noneActive=$inventory.stdout_stderr.Trim() -eq 'No Active Loggers'
$loggerNames=@([regex]::Matches($inventory.stdout_stderr,'(?m)^\s*Logger Name\s*:\s*(.+?)\s*$') | ForEach-Object { $_.Groups[1].Value.Trim() })
$recognizedInventory=$noneActive -or $loggerNames.Count -gt 0
$instanceCollision=@($loggerNames | Where-Object { $_.IndexOf($InstanceName,[StringComparison]::OrdinalIgnoreCase) -ge 0 }).Count -gt 0
$details=InvokeTool 'profile-details' $wpr @('-profiledetails',($profile+'!OwnedThreads'),'-filemode')
NeedSuccess $details
$owner=[ordered]@{schema='llm-sim-etw-broker-1';instance_name=$InstanceName;broker_pid=$PID;broker_start_utc=[Diagnostics.Process]::GetCurrentProcess().StartTime.ToUniversalTime().ToString('o');created_at_utc=[DateTime]::UtcNow.ToString('o');qpc_frequency=[Diagnostics.Stopwatch]::Frequency;maximum_trace_seconds=$MaximumTraceSeconds;capture_requested=[bool]$Capture;current_token_elevated=$isAdmin;profile_privilege_present=$hasProfilePrivilege;no_active_loggers=$noneActive;profile_path=$profile;profile_sha256=(Get-FileHash -LiteralPath $profile -Algorithm SHA256).Hash.ToLowerInvariant();script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant();wpr_path=$wpr;wpr_sha256=(Get-FileHash -LiteralPath $wpr -Algorithm SHA256).Hash.ToLowerInvariant();xperf_path=$xperf;xperf_sha256=(Get-FileHash -LiteralPath $xperf -Algorithm SHA256).Hash.ToLowerInvariant();unrelated_process_filtering_at_capture='not supported; raw system events retained locally; derived output must filter owned lifetime IDs';explicit_global_timer_or_power_configuration_commands_issued=$false}
$owner['logger_inventory_recognized']=$recognizedInventory
$owner['existing_logger_count']=$loggerNames.Count
$owner['own_instance_name_collision']=$instanceCollision
$owner['foreign_loggers_may_coexist']=$true
SaveJson 'preflight.json' $owner
if (-not $Capture) { Write-Output 'Validated only; no recorder or UAC started.'; exit 0 }
$startedOwned=$false
$stopAttempted=$false
$failure=$null
$stopReason='unknown'
function StopOwned {
    $script:stopAttempted=$true
    $marker=InvokeTool 'end-marker' $wpr @('-marker',('LLMSimEnd_'+$InstanceName),'-instancename',$InstanceName)
    $stopResult=InvokeTool 'owned-stop' $wpr @('-stop',$tracePath,'LLM Simulator owned-thread scheduler diagnosis','-skipPdbGen','-instancename',$InstanceName)
    NeedSuccess $stopResult
    SaveJson 'capture-stopped.json' ([ordered]@{schema='llm-sim-etw-stopped-1';instance_name=$InstanceName;reason=$script:stopReason;etl_path=$tracePath;etl_sha256=(Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant();etl_bytes=(Get-Item -LiteralPath $tracePath).Length;end_marker_success=($marker.exit_code -eq 0);qpc_frequency=[Diagnostics.Stopwatch]::Frequency;stopped_at_utc=[DateTime]::UtcNow.ToString('o');exact_scheduler_intervals_validated=$false;lost_events_checked=$false;etl_clock_checked=$false})
}
try {
    if (-not $isAdmin -or -not $hasProfilePrivilege) { throw 'An elevated token containing SeSystemProfilePrivilege is required. No recording started.' }
    if (-not $recognizedInventory) { throw 'Unrecognized logger inventory; recording not started.' }
    if ($instanceCollision) { throw 'Owned instance name already present; recording not started.' }
    $start=InvokeTool 'owned-start' $wpr @('-start',($profile+'!OwnedThreads'),'-filemode','-recordtempto',$captureRoot,'-instancename',$InstanceName)
    NeedSuccess $start
    $startedOwned=$true
    $startTick=$start.qpc_before
    if ([Diagnostics.Stopwatch]::GetTimestamp()-$startTick -ge $MaximumTraceSeconds*[Diagnostics.Stopwatch]::Frequency) { throw 'Recorder startup consumed the capture window; no actor launch is allowed.' }
    $ownStatus=InvokeTool 'owned-status-after-start' $wpr @('-status','profiles','collectors','-details','-instancename',$InstanceName)
    NeedSuccess $ownStatus
    $marker=InvokeTool 'begin-marker' $wpr @('-marker',('LLMSimBegin_'+$InstanceName),'-instancename',$InstanceName)
    NeedSuccess $marker
    SaveJson 'capture-ready.json' ([ordered]@{schema='llm-sim-etw-ready-1';instance_name=$InstanceName;broker_pid=$PID;qpc_ready=[Diagnostics.Stopwatch]::GetTimestamp();qpc_frequency=[Diagnostics.Stopwatch]::Frequency;qpc_deadline=$startTick+$MaximumTraceSeconds*[Diagnostics.Stopwatch]::Frequency;stop_request_path=$stopPath;expected_stop_schema='llm-sim-etw-stop-request-1';actors_may_now_be_launched=$true})
    while ($true) {
        if (Test-Path -LiteralPath $stopPath) {
            $request=Get-Content -LiteralPath $stopPath -Raw | ConvertFrom-Json
            if ($request.schema -ne 'llm-sim-etw-stop-request-1' -or $request.instance_name -ne $InstanceName) { throw 'Stop request identity differs; only this owned recording will be stopped.' }
            $stopReason='matching_controller_request'; break
        }
        if ([Diagnostics.Stopwatch]::GetTimestamp()-$startTick -ge $MaximumTraceSeconds*[Diagnostics.Stopwatch]::Frequency) { $stopReason='hard_trace_deadline'; break }
        Start-Sleep -Milliseconds 20
    }
} catch {
    $failure=$_.Exception.ToString()
    $stopReason='broker_failure'
} finally {
    if ($startedOwned -and -not $stopAttempted) {
        try { StopOwned } catch { if ($null -eq $failure) { $failure=$_.Exception.ToString() } else { $failure+="`nStop failure: "+$_.Exception.ToString() } }
    }
    SaveJson 'broker-final.json' ([ordered]@{schema='llm-sim-etw-broker-final-1';instance_name=$InstanceName;start_confirmed=$startedOwned;stop_attempted=$stopAttempted;failure=$failure;reason=$stopReason;recording_stop_success=(Test-Path -LiteralPath (Join-Path $captureRoot 'capture-stopped.json'));other_recorders_stopped=$false;cancel_command_issued=$false;automatic_symbol_download_requested=$false})
}
if ($null -ne $failure) { throw $failure }
