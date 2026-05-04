param(
    [string]$Ego4DRoot = "E:\datasets\ego4d"
)

$repoDatasetRoot = Resolve-Path ".\datasets" -ErrorAction SilentlyContinue
if (-not $repoDatasetRoot) {
    New-Item -ItemType Directory -Force -Path ".\datasets" | Out-Null
}

$repoEgo4D = ".\datasets\ego4d"
$targetV2 = Join-Path $Ego4DRoot "v2"
$targetAnnotations = Join-Path $targetV2 "annotations"
New-Item -ItemType Directory -Force -Path $targetAnnotations | Out-Null

$datasetsRoot = Resolve-Path ".\datasets"

if ((Test-Path $repoEgo4D) -and -not ((Get-Item $repoEgo4D).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
    $sourceAnnotations = Join-Path $repoEgo4D "v2\annotations"
    if (Test-Path $sourceAnnotations) {
        Copy-Item (Join-Path $sourceAnnotations "*") $targetAnnotations -Force
        Write-Host "Copied existing annotations to $targetAnnotations"
    }

    $backup = Join-Path $datasetsRoot "ego4d_d_drive_backup_$(Get-Date -Format yyyyMMdd_HHmmss)"
    Move-Item -LiteralPath $repoEgo4D -Destination $backup
    Write-Host "Moved existing D-drive Ego4D folder to $backup"
}

if (-not (Test-Path $repoEgo4D)) {
    New-Item -ItemType Junction -Path $repoEgo4D -Target $Ego4DRoot | Out-Null
    Write-Host "Created junction: $repoEgo4D -> $Ego4DRoot"
} else {
    $item = Get-Item $repoEgo4D
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        Write-Host "Junction already exists: $repoEgo4D"
    } else {
        Write-Error "$repoEgo4D exists and is not a junction. Please inspect it before continuing."
        exit 1
    }
}

Write-Host "Ego4D data path for code remains: datasets/ego4d/v2"
Write-Host "Physical storage path is: $targetV2"
