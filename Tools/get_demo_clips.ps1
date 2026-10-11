# Foothold Safety Assistant - download demo footage for a presentation/hackathon
#
# IMPORTANT: these clips are for DEMONSTRATION ONLY.
#   They are NOT training data. They carry no safe/risky/uncertain labels and no
#   trail/date metadata, so using them for training would break RULES.md 4.2,
#   4.4 and 4.8. Use them to show the overlay working, then discard.
#
# Source: Pexels (https://www.pexels.com/license/) - free to use, no attribution
# required (credit is appreciated). Do not resell the clips unaltered or present
# them as your own footage.
#
# Usage:
#   .\get_demo_clips.ps1                      # show the list
#   .\get_demo_clips.ps1 -Download            # fetch them all
#   .\get_demo_clips.ps1 -Download -Dest D:\foothold_demo
#   .\get_demo_clips.ps1 -Download -Only mud,wet
#
# Requires PowerShell execution policy bypass on this machine:
#   powershell -ExecutionPolicy Bypass -File .\get_demo_clips.ps1 -Download
#
# Downloads go to the D: drive by default (the C: drive is nearly full).
# Change with -Dest, or check the target is writable first with -Test.

[CmdletBinding()]
param(
    [switch]$Download,
    [switch]$Test,
    [string]$Dest = "D:\foothold_demo",
    [string[]]$Only = @()
)

$ErrorActionPreference = "Continue"

# id first seen as how the clip shows up in a search, terrain tag we care about,
# and a one-line description of what the shot contains.
$Clips = @(
    @{ Id = 36437881; Tag = "rocky";  Desc = "Hiking a narrow rocky mountain path" }
    @{ Id = 36912295; Tag = "rocky";  Desc = "Hiking adventure on a rocky mountain trail" }
    @{ Id = 5873422;  Tag = "rocky";  Desc = "Person hiking over rough ground" }
    @{ Id = 36372426; Tag = "leaves"; Desc = "Walking through autumn leaves on a forest path" }
    @{ Id = 6960740;  Tag = "leaves"; Desc = "Person walking on dried leaves (ground-level)" }
    @{ Id = 8552296;  Tag = "rocky";  Desc = "Narrow rocky trail with mountain views (vertical)" }
    @{ Id = 12984566; Tag = "mud";    Desc = "Paddles on a wet forest road" }
    @{ Id = 28210606; Tag = "mud";    Desc = "Vehicle crossing a muddy stream/water crossing" }
    @{ Id = 27792994; Tag = "trail";  Desc = "Wilderness road/path into forest" }
    @{ Id = 8865474;  Tag = "feet";   Desc = "Walking feet (close-up footsteps on ground)" }
    @{ Id = 4085318;  Tag = "trail";  Desc = "Walking in a park along a path" }
    @{ Id = 5597043;  Tag = "trail";  Desc = "Person walking outdoors on a path" }
    @{ Id = 34400403; Tag = "trail";  Desc = "Walking forward into foliage" }
    @{ Id = 19848529; Tag = "trail";  Desc = "Forest path (ground-level walk)" }
    @{ Id = 7157038;  Tag = "trail";  Desc = "Forest trail walk" }
)

$Selected = if ($Only.Count -gt 0) {
    $Clips | Where-Object { $Only -contains $_.Tag }
} else { $Clips }

function Show-Clips {
    Write-Host ""
    Write-Host "Demo clips (Pexels, free to use) - DEMONSTRATION ONLY, not training data" -ForegroundColor Cyan
    Write-Host ("=" * 100)
    Write-Host ("  {0,-10} {1,-8} {2}" -f "ID", "TAG", "CONTENT") -ForegroundColor Yellow
    foreach ($c in $Clips) {
        Write-Host ("  {0,-10} {1,-8} {2}" -f $c.Id, $c.Tag, $c.Desc)
    }
    Write-Host ("=" * 100)
    $tags = ($Clips | ForEach-Object { $_.Tag } | Sort-Object -Unique) -join ", "
    Write-Host ("  {0} clips. Tags: {1}" -f $Clips.Count, $tags)
    Write-Host "  Download to: $Dest"
    Write-Host ""
    Write-Host "  Direct link format (works without login):"
    Write-Host "    https://www.pexels.com/download/video/<ID>/"
    Write-Host "  or build the direct file URL from a video page, e.g.:"
    Write-Host "    https://videos.pexels.com/video-files/8552296/8552296-hd_1080_1920_30fps.mp4"
    Write-Host ""
}

if ($Test) {
    Write-Host ""
    Write-Host "Testing destination: $Dest" -ForegroundColor Cyan
    $TestPassed = $true
    try {
        if (-not (Test-Path -LiteralPath $Dest)) {
            New-Item -ItemType Directory -Force -Path $Dest -ErrorAction Stop | Out-Null
            Write-Host "  created the folder" -ForegroundColor Green
        } else {
            Write-Host "  folder already exists" -ForegroundColor Green
        }
        $probe = Join-Path $Dest ".write_test"
        "ok" | Out-File -FilePath $probe -Encoding ascii -ErrorAction Stop
        Remove-Item $probe -Force -ErrorAction Stop
        Write-Host "  write + delete test passed" -ForegroundColor Green

        $drive = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath $Dest).Path)
        $di = New-Object System.IO.DriveInfo($drive)
        $freeGB = [math]::Round($di.AvailableFreeSpace / 1GB, 1)
        Write-Host "  drive $drive has $freeGB GB free"
        if ($freeGB -lt 2) {
            Write-Host "  WARNING: less than 2 GB free - 15 clips need roughly 0.5-2 GB." -ForegroundColor Yellow
        }
    } catch {
        $TestPassed = $false
        Write-Host "  FAILED: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        Write-Host "  The folder could not be created or written to." -ForegroundColor Yellow
        Write-Host "  Create it manually, then re-run -Test:" -ForegroundColor Yellow
        Write-Host "    New-Item -ItemType Directory -Force -Path `"$Dest`""
    }
    Write-Host ""
    if ($TestPassed) {
        Write-Host "Destination is writable. Run with -Download to fetch the clips." -ForegroundColor Green
    }
    return
}

if (-not $Download) {
    Show-Clips
    Write-Host "Re-run with -Download to fetch them, or -Test to check the target folder." -ForegroundColor Green
    return
}

if (-not (Test-Path -LiteralPath $Dest)) {
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    Write-Host "created $Dest"
}

Write-Host ""
Write-Host "Downloading $($Selected.Count) clips to $Dest" -ForegroundColor Cyan
Write-Host ""

$ok = 0; $fail = 0
foreach ($c in $Selected) {
    $out = Join-Path $Dest ("pexels_{0}_{1}.mp4" -f $c.Id, $c.Tag)
    if ((Test-Path -LiteralPath $out) -and (Get-Item $out).Length -gt 10000) {
        Write-Host ("  skip   {0} (already have it)" -f $c.Id) -ForegroundColor DarkGray
        $ok++
        continue
    }
    Write-Host ("  fetch  {0}  {1}" -f $c.Id, $c.Desc)
    # -L follows the 302 to videos.pexels.com. curl.exe ships with Windows.
    & curl.exe -sS -L --fail --retry 3 --retry-delay 2 -o $out `
        -A "Mozilla/5.0" "https://www.pexels.com/download/video/$($c.Id)/" 2>$null

    if ($LASTEXITCODE -eq 0 -and (Test-Path $out) -and (Get-Item $out).Length -gt 10000) {
        $mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
        Write-Host ("         -> {0} MB" -f $mb) -ForegroundColor Green
        $ok++
    } else {
        Write-Host "         -> FAILED (try opening the page in a browser)" -ForegroundColor Yellow
        Write-Host ("            https://www.pexels.com/video/{0}/" -f $c.Id)
        if (Test-Path $out) { Remove-Item $out -Force }
        $fail++
    }
}

Write-Host ""
Write-Host ("Done. {0} downloaded, {1} failed." -f $ok, $fail)
Write-Host ""
Write-Host "Verify each file actually decodes before presenting:" -ForegroundColor Cyan
Write-Host "  `$py = 'C:\Users\mriga\.venv\Scripts\python.exe'"
Write-Host "  Get-ChildItem '$Dest\*.mp4' | ForEach-Object { & `$py tools\make_test_video.py info `$_.FullName }"
Write-Host ""
Write-Host "Then run the app against any of them:"
Write-Host "  & `$py src\qt_app.py --source `"$Dest\pexels_36437881_rocky.mp4`""
