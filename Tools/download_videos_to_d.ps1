# Foothold Safety Assistant - download the demo clips straight to the D: drive
#
# Why this exists: if you click a download link in a browser it lands in the
# browser's default folder (C:\Users\mriga\Downloads), not on D:. This script
# fetches the files directly with curl and writes them wherever you say, so the
# browser and its download folder are never involved.
#
# These clips are for the HACKATHON DEMO ONLY. They carry no safe/risky/uncertain
# labels and no trail metadata, so they must NOT be used for training
# (see RULES.md 4.2, 4.4 and 4.8).
#
# Source: Pexels - free to use, no attribution required.
#
# Usage:
#   .\download_videos_to_d.ps1                 # show the plan
#   .\download_videos_to_d.ps1 -Go             # download to D:\foothold_demo
#   .\download_videos_to_d.ps1 -Go -Dest "D:\videos\hackathon"
#   .\download_videos_to_d.ps1 -Go -Only rocky,mud

[CmdletBinding()]
param(
    [switch]$Go,
    [string]$Dest = "D:\foothold_demo",
    [string[]]$Only = @()
)

$ErrorActionPreference = "Continue"

# Id, tag, and a short description of the shot.
$Clips = @(
    @{ Id = 36437881; Tag = "rocky";  Desc = "Narrow rocky mountain path" }
    @{ Id = 36912295; Tag = "rocky";  Desc = "Rocky mountain trail" }
    @{ Id = 5873422;  Tag = "rocky";  Desc = "Hiking over rough ground" }
    @{ Id = 8552296;  Tag = "rocky";  Desc = "Rocky trail, mountain views (VERTICAL 1080x1920)" }
    @{ Id = 12984566; Tag = "mud";    Desc = "Wet forest road" }
    @{ Id = 28210606; Tag = "mud";    Desc = "Muddy water crossing (vehicle, not on foot)" }
    @{ Id = 36372426; Tag = "leaves"; Desc = "Autumn leaves on a forest path" }
    @{ Id = 6960740;  Tag = "leaves"; Desc = "Footsteps on dried leaves" }
    @{ Id = 8865474;  Tag = "feet";   Desc = "Close-up walking feet" }
    @{ Id = 19848529; Tag = "trail";  Desc = "Forest path, ground-level walk" }
    @{ Id = 7157038;  Tag = "trail";  Desc = "Forest trail walk" }
    @{ Id = 4085318;  Tag = "trail";  Desc = "Walking a park path" }
    @{ Id = 5597043;  Tag = "trail";  Desc = "Person walking outdoors" }
    @{ Id = 34400403; Tag = "trail";  Desc = "Walking into foliage" }
    @{ Id = 27792994; Tag = "trail";  Desc = "Wilderness path into forest" }
)

$Sel = if ($Only.Count -gt 0) { $Clips | Where-Object { $Only -contains $_.Tag } } else { $Clips }

# --- Where browsers currently put things (shown for contrast) ---
$browserDl = Join-Path $env:USERPROFILE "Downloads"

Write-Host ""
Write-Host "Download destination" -ForegroundColor Cyan
Write-Host ("=" * 78)
Write-Host "  Browser default : $browserDl   <-- C: drive, nearly full"
Write-Host "  This script     : $Dest" -ForegroundColor Green
Write-Host ("=" * 78)
Write-Host ""

if (-not $Go) {
    Write-Host ("  {0,-10} {1,-8} {2}" -f "ID", "TAG", "SHOT") -ForegroundColor Yellow
    foreach ($c in $Sel) { Write-Host ("  {0,-10} {1,-8} {2}" -f $c.Id, $c.Tag, $c.Desc) }
    Write-Host ""
    Write-Host "  $($Sel.Count) clips selected." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  These will be saved as:"
    Write-Host "    $Dest\pexels_<ID>_<tag>.mp4"
    Write-Host ""
    Write-Host "  Re-run with -Go to download:" -ForegroundColor Green
    Write-Host "    powershell -ExecutionPolicy Bypass -File .\tools\download_videos_to_d.ps1 -Go"
    Write-Host ""
    return
}

# --- Make sure the target exists and is actually writable ---
try {
    if (-not (Test-Path -LiteralPath $Dest)) {
        New-Item -ItemType Directory -Force -Path $Dest -ErrorAction Stop | Out-Null
        Write-Host "created $Dest" -ForegroundColor Green
    }
    $probe = Join-Path $Dest ".write_test"
    "ok" | Out-File -FilePath $probe -Encoding ascii -ErrorAction Stop
    Remove-Item $probe -Force -ErrorAction Stop
} catch {
    Write-Host ""
    Write-Host "Cannot write to '$Dest': $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "Create it yourself, then re-run:" -ForegroundColor Yellow
    Write-Host "  New-Item -ItemType Directory -Force -Path `"$Dest`""
    Write-Host ""
    exit 1
}

$drive = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Dest).Path)
$freeGB = [math]::Round((New-Object System.IO.DriveInfo($drive)).AvailableFreeSpace / 1GB, 1)
Write-Host "Free on $drive : $freeGB GB" -ForegroundColor Cyan
Write-Host ""

$ok = 0; $skip = 0; $bad = 0
foreach ($c in $Sel) {
    $out = Join-Path $Dest ("pexels_{0}_{1}.mp4" -f $c.Id, $c.Tag)

    if ((Test-Path -LiteralPath $out) -and (Get-Item $out).Length -gt 10000) {
        Write-Host ("  skip  {0}  (already downloaded)" -f $c.Id) -ForegroundColor DarkGray
        $skip++
        continue
    }

    Write-Host ("  get   {0}  {1}" -f $c.Id, $c.Desc)
    & curl.exe -sS -L --fail --retry 3 --retry-delay 2 -A "Mozilla/5.0" `
        -o $out "https://www.pexels.com/download/video/$($c.Id)/" 2>$null

    if ($LASTEXITCODE -eq 0 -and (Test-Path $out) -and (Get-Item $out).Length -gt 10000) {
        Write-Host ("        -> {0} MB" -f [math]::Round((Get-Item $out).Length / 1MB, 1)) -ForegroundColor Green
        $ok++
    } else {
        Write-Host "        -> FAILED - open the page in a browser instead:" -ForegroundColor Yellow
        Write-Host ("           https://www.pexels.com/download/video/{0}/" -f $c.Id)
        if (Test-Path $out) { Remove-Item $out -Force }
        $bad++
    }
}

Write-Host ""
Write-Host ("Done: {0} downloaded, {1} already present, {2} failed." -f $ok, $skip, $bad) -ForegroundColor Cyan
Write-Host "Location: $Dest"
Write-Host ""
Write-Host "Verify each decodes, then run the app:" -ForegroundColor Cyan
Write-Host "  `$py = 'C:\Users\mriga\.venv\Scripts\python.exe'"
Write-Host "  Get-ChildItem `"$Dest\*.mp4`" | ForEach-Object { & `$py tools\make_test_video.py info `$_.FullName }"
Write-Host "  & `$py src\qt_app.py --source `"$Dest\pexels_36437881_rocky.mp4`""
