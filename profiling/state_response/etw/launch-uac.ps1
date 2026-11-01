[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Broker,
    [Parameter(Mandatory=$true)][string]$OutDir,
    [Parameter(Mandatory=$true)][string]$InstanceName,
    [Parameter(Mandatory=$true)][string]$Receipt
)
$ErrorActionPreference='Stop'
function Quoted([string]$Value) {
    if ($Value.Contains('"')) { throw 'Unexpected quote in a controlled path.' }
    return '"'+$Value+'"'
}
$hostExecutable=(Get-Process -Id $PID).Path
$arguments=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',(Quoted $Broker),'-OutDir',(Quoted $OutDir),'-InstanceName',$InstanceName,'-MaximumTraceSeconds','30','-Capture')
$started=Start-Process -FilePath $hostExecutable -ArgumentList $arguments -Verb RunAs -WindowStyle Hidden -PassThru
$row=[ordered]@{schema='llm-sim-etw-uac-launch-1';broker_pid=$started.Id;instance_name=$InstanceName;out_dir=$OutDir;created_at_utc=[DateTime]::UtcNow.ToString('o');window_style='Hidden';maximum_trace_seconds=30}
$text=($row | ConvertTo-Json -Depth 4).Replace("`r`n","`n")+"`n"
$file=[IO.File]::Open($Receipt,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try { $bytes=(New-Object Text.UTF8Encoding($false)).GetBytes($text);$file.Write($bytes,0,$bytes.Length) } finally { $file.Dispose() }
