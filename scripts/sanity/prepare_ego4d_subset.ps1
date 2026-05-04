$env:TOKENIZERS_PARALLELISM = "false"
$python = "D:\Users\15373\anaconda3\envs\video-mm\python.exe"

& $python .\scripts\sanity\select_ego4d_narration_subset.py `
    --train_annotation datasets/ego4d/v2/annotations/refined_narration_stream_train.json `
    --val_annotation datasets/ego4d/v2/annotations/refined_narration_stream_val.json `
    --output_dir outputs/ego4d_subset `
    --train_count 10 `
    --val_count 3 `
    --min_narrations 20 `
    --max_duration 600 `
    --prefer_existing_video_dir datasets/ego4d/v2/full_scale
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Next download the selected videos with Ego4D CLI if they are not already present:"
Write-Host ".\scripts\sanity\download_ego4d_subset.ps1"
