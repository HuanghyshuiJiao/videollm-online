param(
    [string]$Source = "outputs/ego4d_credentials.ini",
    [string]$Profile = "default",
    [switch]$Force
)

$awsDir = Join-Path $env:USERPROFILE ".aws"
$target = Join-Path $awsDir "credentials"

if (-not (Test-Path $Source)) {
    New-Item -ItemType Directory -Force -Path (Split-Path $Source) | Out-Null
    Copy-Item .\scripts\sanity\ego4d_credentials.template.ini $Source
    Write-Host "Created $Source"
    Write-Host "Fill in your Ego4D AWS keys there, then run this script again."
    exit 0
}

$content = Get-Content $Source -Raw
if ($content -match "PASTE_YOUR_EGO4D") {
    Write-Error "Please replace the placeholders in $Source first."
    exit 1
}

New-Item -ItemType Directory -Force -Path $awsDir | Out-Null
if ((Test-Path $target) -and -not $Force) {
    Write-Error "$target already exists. Re-run with -Force if you want to overwrite it."
    exit 1
}

Copy-Item $Source $target -Force
Write-Host "Installed Ego4D AWS credentials to $target"
Write-Host "Using AWS profile: $Profile"

