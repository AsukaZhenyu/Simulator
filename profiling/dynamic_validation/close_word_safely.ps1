param(
    [Parameter(Mandatory=$true)][ValidateSet('Inspect','Close')][string]$Mode,
    [Parameter(Mandatory=$true)][string]$Out,
    [Parameter(Mandatory=$true)][string]$RecoveryDirectory
)
$ErrorActionPreference='Stop'
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
public static class ExistingOfficeObjects {
    [DllImport("ole32.dll")] static extern int GetRunningObjectTable(int reserved, out IRunningObjectTable table);
    [DllImport("ole32.dll")] static extern int CreateBindCtx(int reserved, out IBindCtx context);
    [DllImport("ole32.dll", CharSet=CharSet.Unicode)] static extern int CLSIDFromProgID(string name, out Guid id);
    [DllImport("oleaut32.dll")] static extern int GetActiveObject(ref Guid clsid, IntPtr reserved, [MarshalAs(UnmanagedType.IUnknown)] out object instance);
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr window, out uint processId);
    public static object[] Get() {
        var items=new List<object>(); Guid word; object active;
        if (CLSIDFromProgID("Word.Application",out word)>=0 && GetActiveObject(ref word,IntPtr.Zero,out active)>=0) items.Add(active);
        IRunningObjectTable table; IBindCtx context; IEnumMoniker cursor;
        if (GetRunningObjectTable(0,out table)<0 || CreateBindCtx(0,out context)<0) return items.ToArray();
        table.EnumRunning(out cursor); var one=new IMoniker[1];
        try {
            while(cursor.Next(1,one,IntPtr.Zero)==0) {
                try {
                    string name; one[0].GetDisplayName(context,null,out name);
                    string lower=name.ToLowerInvariant();
                    if (!(lower.EndsWith(".docx")||lower.EndsWith(".docm")||lower.EndsWith(".doc")||lower.EndsWith(".dotx")||lower.EndsWith(".dotm"))) continue;
                    object item; table.GetObject(one[0],out item); if(item!=null) items.Add(item);
                } catch {} finally { if(one[0]!=null) Marshal.ReleaseComObject(one[0]); }
            }
        } finally { Marshal.ReleaseComObject(cursor); Marshal.ReleaseComObject(context); Marshal.ReleaseComObject(table); }
        return items.ToArray();
    }
}
'@
$applications=@{}
foreach($item in [ExistingOfficeObjects]::Get()) {
    try {
        $app=$item.Application
        if($app.Name -ne 'Microsoft Word') {continue}
        $identity=[Runtime.InteropServices.Marshal]::GetIUnknownForObject($app)
        $key=$identity.ToInt64();[void][Runtime.InteropServices.Marshal]::Release($identity)
        if(-not $applications.ContainsKey($key)) {$applications[$key]=$app}
    } catch {}
}
$recovery=[IO.Path]::GetFullPath($RecoveryDirectory)
[void][IO.Directory]::CreateDirectory($recovery)
$records=[Collections.Generic.List[object]]::new();$index=0
foreach($app in $applications.Values) {
    $index++;$row=[ordered]@{instance=$index;status='inspected';documents=@();quit_requested=$false}
    try {
        $row.visible=[bool]$app.Visible;$row.hwnd=[long]$app.Hwnd
        [uint32]$ownerPid=0
        if($row.hwnd -ne 0) {[void][ExistingOfficeObjects]::GetWindowThreadProcessId([IntPtr]$row.hwnd,[ref]$ownerPid)}
        $row.pid=$ownerPid
        $documents=@();for($n=1;$n -le $app.Documents.Count;$n++) {$documents+=,$app.Documents.Item($n)}
        foreach($doc in $documents) {
            $entry=[ordered]@{name=[string]$doc.Name;path=[string]$doc.Path;saved=[bool]$doc.Saved;recovery_path=$null}
            if($Mode -eq 'Close' -and -not $doc.Saved) {
                $macro=[bool]$doc.HasVBProject
                $extension=if($macro){'.docm'}else{'.docx'};$format=if($macro){13}else{16}
                $copy=Join-Path $recovery ('Word_'+[DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss_fff')+'_'+$index+'_'+$n+$extension)
                if(Test-Path -LiteralPath $copy) {throw 'recovery target already exists'}
                $doc.SaveAs2($copy,$format)
                if(-not $doc.Saved -or -not (Test-Path -LiteralPath $copy)) {throw 'document recovery save not confirmed'}
                $entry.recovery_path=$copy;$entry.saved=$true
            }
            $row.documents+=,[pscustomobject]$entry
        }
        if($Mode -eq 'Close') {
            foreach($doc in $documents) {if(-not $doc.Saved){throw 'unsaved document remains; keep this Word instance open'}}
            $app.Quit(0);$row.quit_requested=$true;$row.status='saved_or_already_saved_then_quit'
        }
    } catch {$row.status='kept_open';$row.error=$_.Exception.Message}
    $records.Add([pscustomobject]$row)
}
$result=[ordered]@{schema='safe-existing-word-release-1';mode=$Mode;recorded_at_utc=[DateTime]::UtcNow.ToString('o');
    new_Word_instance_created=$false;force_terminated=$false;recovery_directory=$recovery;instances=$records}
$absolute=[IO.Path]::GetFullPath($Out)
if(Test-Path -LiteralPath $absolute) {throw 'receipt already exists'}
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($absolute))
[IO.File]::WriteAllText($absolute,(($result|ConvertTo-Json -Depth 9)-replace "`r`n","`n")+"`n",[Text.UTF8Encoding]::new($false))
[pscustomobject]@{receipt=$absolute;instances=$records.Count;quit_requested=@($records|Where-Object quit_requested -eq $true).Count;
    kept_open=@($records|Where-Object status -eq 'kept_open').Count;recovery_copies=@($records.documents|Where-Object recovery_path -ne $null).Count}|ConvertTo-Json -Compress
