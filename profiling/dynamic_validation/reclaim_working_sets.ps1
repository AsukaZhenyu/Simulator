param(
    [Parameter(Mandatory=$true)][ValidateSet('Background','BrowserHelpers','AppCaches','ShellCaches')][string]$Mode,
    [Parameter(Mandatory=$true)][UInt64]$TargetAvailableBytes,
    [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference = 'Stop'
# EmptyWorkingSet evicts resident pages through Windows. It does not terminate
# processes, close documents, save files, or impose persistent working-set caps.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class OwnedSessionMemoryTrim {
    [StructLayout(LayoutKind.Sequential)] public struct Status {
        public uint Length, Load;
        public ulong Total, Available, PageTotal, PageAvailable, VirtualTotal, VirtualAvailable, Extended;
    }
    [DllImport("kernel32.dll", SetLastError=true)] static extern bool GlobalMemoryStatusEx(ref Status s);
    [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
    [DllImport("kernel32.dll", SetLastError=true)] public static extern bool CloseHandle(IntPtr handle);
    [DllImport("psapi.dll", SetLastError=true)] public static extern bool EmptyWorkingSet(IntPtr handle);
    public static ulong AvailableBytes() {
        Status s = new Status(); s.Length = (uint)Marshal.SizeOf(typeof(Status));
        if (!GlobalMemoryStatusEx(ref s)) throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());
        return s.Available;
    }
}
'@
$session = [System.Diagnostics.Process]::GetCurrentProcess().SessionId
$before = [OwnedSessionMemoryTrim]::AvailableBytes()
$names = if ($Mode -eq 'Background') {
    @('NVIDIA Overlay','Weixin','WeChatAppEx','PhoneExperienceHost','WINWORD','POWERPNT','EXCEL','chrome','msedge','firefox','Acrobat','AcroRd32')
} elseif ($Mode -eq 'BrowserHelpers') { @('ChatGPT') }
elseif ($Mode -eq 'ShellCaches') { @('explorer','SearchHost','TextInputHost','StartMenuExperienceHost','ShellExperienceHost','Widgets','msedgewebview2','ApplicationFrameHost') }
else { @('ChatGPT','codex') }
$candidates = @(Get-Process | Where-Object {
    $_.SessionId -eq $session -and $_.Id -ne $PID -and
    $names -contains $_.ProcessName -and $_.WorkingSet64 -gt $(if ($Mode -eq 'AppCaches') {192MB} else {32MB}) -and
    ($Mode -ne 'BrowserHelpers' -or $_.MainWindowHandle -eq [IntPtr]::Zero)
} | Sort-Object WorkingSet64 -Descending)
$records = [System.Collections.Generic.List[object]]::new()
foreach ($candidate in $candidates) {
    if ([OwnedSessionMemoryTrim]::AvailableBytes() -ge $TargetAvailableBytes) { break }
    $handle = [IntPtr]::Zero
    $row = [ordered]@{pid=$candidate.Id; name=$candidate.ProcessName; session_id=$session;
        resident_before_bytes=$candidate.WorkingSet64; status='not_attempted'; application_closed=$false}
    try {
        $fresh = Get-Process -Id $candidate.Id -ErrorAction Stop
        if ($fresh.SessionId -ne $session -or $fresh.ProcessName -ne $candidate.ProcessName) { throw 'process identity changed' }
        $handle = [OwnedSessionMemoryTrim]::OpenProcess(0x0500, $false, [uint32]$fresh.Id)
        if ($handle -eq [IntPtr]::Zero) { throw [System.ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
        if (-not [OwnedSessionMemoryTrim]::EmptyWorkingSet($handle)) { throw [System.ComponentModel.Win32Exception]::new([Runtime.InteropServices.Marshal]::GetLastWin32Error()) }
        $fresh.Refresh(); $row.resident_after_bytes=$fresh.WorkingSet64; $row.status='trimmed'
    } catch { $row.status='unavailable'; $row.error=$_.Exception.Message }
    finally { if ($handle -ne [IntPtr]::Zero) { [void][OwnedSessionMemoryTrim]::CloseHandle($handle) } }
    $records.Add([pscustomobject]$row)
}
$after = [OwnedSessionMemoryTrim]::AvailableBytes()
$result = [ordered]@{
    schema='same-session-working-set-reclaim-1'; recorded_at_utc=[DateTime]::UtcNow.ToString('o');
    mode=$Mode; available_before_bytes=$before; available_after_bytes=$after;
    target_available_bytes=$TargetAvailableBytes; applications_closed=$false;
    documents_saved_or_modified=$false; persistent_memory_limits_changed=$false;
    protected_system_security_network_processes_targeted=$false;
    agent_application_working_set_targeted=($Mode -eq 'AppCaches');
    user_session_shell_cache_targeted=($Mode -eq 'ShellCaches');
    processes=$records; note='Resident pages may return on later application access. Recheck memory before inference; this is not a performance calibration.'
}
$absolute = [System.IO.Path]::GetFullPath($Out)
if (Test-Path -LiteralPath $absolute) { throw 'output receipt already exists' }
[void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($absolute))
$json = $result | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($absolute, ($json -replace "`r`n","`n")+"`n", [System.Text.UTF8Encoding]::new($false))
[pscustomobject]@{receipt=$absolute; available_before_bytes=$before; available_after_bytes=$after;
    trimmed=@($records | Where-Object status -eq 'trimmed').Count; failed=@($records | Where-Object status -eq 'unavailable').Count;
    applications_closed=$false} | ConvertTo-Json -Compress
