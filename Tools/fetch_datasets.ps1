# Foothold Safety Assistant - dataset download helper
#
# Resumable downloads for the public terrain datasets, using only tooling that
# already ships with Windows (curl + tar). Nothing here talks to chat or email,
# per RULES.md 6.8 - data lives on a local/shared drive.
#
# Usage:
#   .\fetch_datasets.ps1 -List
#   .\fetch_datasets.ps1 -Dataset ycor
#   .\fetch_datasets.ps1 -Dataset rellis-images
#   .\fetch_datasets.ps1 -Dataset ycor -Dest D:\foothold_datasets
#
# NOTE: a browser download is often faster and more reliable for Google Drive
# and Box, because both rate-limit and show a virus-scan interstitial. This
# script is for the direct links and for resume/verify convenience.

[CmdletBinding()]
param(
    [string]$Dataset = "",
    [string]$Dest = "D:\foothold_datasets",
    [switch]$List,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# name = @{ Url; SizeGB (approx); Note; Kind }
$Datasets = [ordered]@{
    "ycor" = @{
        Url    = "https://cmu.box.com/s/3fngoljhcwhqf2z5cbepufh331qtesxt"
        SizeGB = 2
        Kind   = "browser"
        Note   = "Yamaha-CMU Off-Road. 1076 imgs, 8 classes (rough/smooth trail...). CC BY 4.0. Box link - browser is easier."
    }
    "rellis-images" = @{
        Url    = "https://drive.google.com/file/d/1F3Leu0H_m6aPVpZITragfreO_SGtL2yV/view"
        SizeGB = 11
        Kind   = "gdown"
        Note   = "RELLIS-3D RGB images only (6235 imgs). This is the one you want - skip the LiDAR/ROS bags."
    }
    "rellis-labels" = @{
        Url    = "https://drive.google.com/file/d/1HJl8Fi5nAjOr41DPUFmkeKWtDXhCZDke/view"
        SizeGB = 0.12
        Kind   = "gdown"
        Note   = "RELLIS-3D colour segmentation masks for the images above."
    }
    "rellis-labels-id" = @{
        Url    = "https://drive.google.com/file/d/16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5/view"
        SizeGB = 0.094
        Kind   = "gdown"
        Note   = "RELLIS-3D ID-format masks (easier to remap to safe/risky/uncertain)."
    }
    "rugd" = @{
        Url    = "https://rugd.vision/"
        SizeGB = 3
        Kind   = "browser"
        Note   = "RUGD, unstructured ground. Use robots-dataset-toolkit to fetch."
    }
    "orfd" = @{
        Url    = "https://github.com/chaytonmin/Off-Road-Freespace-Detection"
        SizeGB = 6
        Kind   = "browser"
        Note   = "ORFD. SEQUENCES (video) + LiDAR. sunny/rainy/foggy/snowy. Apache 2.0."
    }
    "sanpo" = @{
        Url    = "https://console.cloud.google.com/storage/browser/gresearch/sanpo_dataset/v0"
        SizeGB = 12
        Kind   = "gcloud"
        Note   = "VIDEO-FIRST, human egocentric at chest/eye level with real runners. CC BY 4.0 (commercial OK), PII already blurred. WARNING: full set is ~6TB - download individual sessions only."
    }
    "orad3d" = @{
        Url    = "https://modelscope.cn/datasets/Georgehhhh/ORAD-3D"
        SizeGB = 350
        Kind   = "browser"
        Note   = "ORAD-3D. Largest off-road set. Has puddles/snow/rock/safe-road classes. Apache 2.0. 350GB - only if you have the space."
    }
}

function Show-List {
    Write-Host ""
    Write-Host "Available datasets" -ForegroundColor Cyan
    Write-Host ("=" * 96)
    foreach ($k in $Datasets.Keys) {
        $d = $Datasets[$k]
        Write-Host ("  {0,-18} ~{1,5} GB  [{2}]" -f $k, $d.SizeGB, $d.Kind) -ForegroundColor Yellow
        Write-Host ("      {0}" -f $d.Note)
    }
    Write-Host ("=" * 96)
    Write-Host "  Download to: $Dest"
    Write-Host "  Tip: keep big data OFF the C: drive. Check free space first."
    Write-Host ""
}

function Get-FreeGB([string]$Path) {
    $root = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath (Split-Path -Parent $Path) -ErrorAction SilentlyContinue).Path)
    if (-not $root) { $root = [System.IO.Path]::GetPathRoot($Path) }
    if (-not $root) { $root = "C:\" }
    $di = New-Object System.IO.DriveInfo($root)
    return [math]::Round($di.AvailableFreeSpace / 1GB, 1)
}

if ($List -or -not $Dataset) {
    Show-List
    if (-not $Dataset) { return }
}

if (-not $Datasets.Contains($Dataset)) {
    Write-Host "Unknown dataset '$Dataset'." -ForegroundColor Red
    Show-List
    exit 2
}

$meta = $Datasets[$Dataset]
$dlDir = Join-Path $Dest "_downloads"
$exDir = Join-Path $Dest "extracted"

foreach ($p in @($Dest, $dlDir, $exDir)) {
    if (-not (Test-Path -LiteralPath $p)) {
        New-Item -ItemType Directory -Force -Path $p | Out-Null
        Write-Host "created $p"
    }
}

$free = Get-FreeGB $Dest
Write-Host ""
Write-Host "Dataset : $Dataset"
Write-Host "Approx  : ~$($meta.SizeGB) GB"
Write-Host "Dest    : $Dest"
Write-Host "Free    : $free GB"
Write-Host ""

if ($free -lt ($meta.SizeGB * 1.5)) {
    Write-Host "WARNING: free space is less than 1.5x the dataset size." -ForegroundColor Yellow
    Write-Host "         Extraction needs room for BOTH the archive and its contents." -ForegroundColor Yellow
    if (-not $Force) {
        $ans = Read-Host "Continue anyway? (y/N)"
        if ($ans -notmatch '^[Yy]') { Write-Host "aborted."; return }
    }
}

Write-Host $meta.Note
Write-Host ""

switch ($meta.Kind) {
    "browser" {
        Write-Host "This one is best fetched in a browser (login / interstitial / rate limits)." -ForegroundColor Cyan
        Write-Host "Opening: $($meta.Url)"
        Start-Process $meta.Url
        Write-Host ""
        Write-Host "When it lands in Downloads, move it to: $dlDir"
        Write-Host "Then extract with:  tar -xf <archive> -C `"$exDir`""
    }
    "gdown" {
        $venvPy = "C:\Users\mriga\.venv\Scripts\python.exe"
        $py = if (Test-Path $venvPy) { $venvPy } else { "python" }
        Write-Host "Google Drive file - using gdown (handles the confirm token)." -ForegroundColor Cyan
        & $py -c "import gdown" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "gdown not installed. Installing into the venv..."
            & $py -m pip install --quiet gdown
        }
        # Resumable: gdown skips the download if the file is already complete.
        & $py -m gdown --fuzzy $meta.Url -O $dlDir --continue
        if ($LASTEXITCODE -ne 0) {
            Write-Host "gdown failed. Fall back to the browser, or configure rclone." -ForegroundColor Yellow
            Write-Host "Browser: $($meta.Url)"
            exit 1
        }
        Write-Host ""
        Write-Host "Downloaded into $dlDir" -ForegroundColor Green
        Write-Host "Inspect, then extract:  tar -xf <archive> -C `"$exDir`""
    }
    "gcloud" {
        Write-Host "SANPO lives in a Google Cloud Storage bucket - use gcloud, NOT the console." -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Do NOT sync the whole thing (it is ~6 TB). Pull individual sessions:" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  gcloud storage cp -r `"gs://gresearch/sanpo_dataset/v0/sanpo-real/<session>`" `"$exDir`""
        Write-Host ""
        Write-Host "Browse what exists first (session attributes list which are park/suburban):"
        Write-Host "  gcloud storage ls `"gs://gresearch/sanpo_dataset/v0/sanpo-real/`""
        Write-Host ""
        Write-Host "Sanity check the size of ONE session before committing:"
        Write-Host "  gcloud storage du -s -h `"gs://gresearch/sanpo_dataset/v0/sanpo-real/<session>`""
        Write-Host ""
        Write-Host "Aim for a handful of park/suburban sessions - that is all the video" -ForegroundColor DarkGray
        Write-Host "pipeline needs for testing." -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "Then convert PNG frame sequences to mp4 for the app:"
        Write-Host "  $venv = 'C:\Users\mriga\.venv\Scripts\python.exe'"
        Write-Host "  & $venv tools\make_test_video.py frames-to-video <frame_dir> -o data\sanpo_park.mp4 --fps 10 --limit 900"
    }
}

Write-Host ""
Write-Host "Reminder (RULES.md 4.4): split by trail/trip, never randomly by image." -ForegroundColor DarkGray
Write-Host "These datasets are for PRE-TRAINING only - your safe/risky/uncertain" -ForegroundColor DarkGray
Write-Host "labels must stay hand-made from the trekker's chest-height viewpoint." -ForegroundColor DarkGray
