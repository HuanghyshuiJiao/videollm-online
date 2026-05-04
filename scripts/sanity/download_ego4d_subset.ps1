param(
    [string]$Ego4DRoot = "E:\datasets\ego4d",
    [string]$AwsProfile = "default"
)

$python = "D:\Users\15373\anaconda3\envs\video-mm\python.exe"

if (-not (Test-Path outputs\ego4d_subset\train_video_uids.txt)) {
    Write-Error "Missing outputs\ego4d_subset\train_video_uids.txt. Run .\scripts\sanity\prepare_ego4d_subset.ps1 first."
    exit 1
}
if (-not (Test-Path outputs\ego4d_subset\val_video_uids.txt)) {
    Write-Error "Missing outputs\ego4d_subset\val_video_uids.txt. Run .\scripts\sanity\prepare_ego4d_subset.ps1 first."
    exit 1
}

& $python -m ego4d.cli.cli `
    --output_directory $Ego4DRoot `
    --datasets full_scale `
    --version v2 `
    --aws_profile_name $AwsProfile `
    --video_uid_file outputs\ego4d_subset\train_video_uids.txt `
    -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m ego4d.cli.cli `
    --output_directory $Ego4DRoot `
    --datasets full_scale `
    --version v2 `
    --aws_profile_name $AwsProfile `
    --video_uid_file outputs\ego4d_subset\val_video_uids.txt `
    -y
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Downloaded Ego4D subset videos to $Ego4DRoot\v2\full_scale"
