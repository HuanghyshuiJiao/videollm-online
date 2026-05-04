$env:TOKENIZERS_PARALLELISM = "false"
$python = "D:\Users\15373\anaconda3\envs\video-mm\python.exe"

& $python .\scripts\sanity\prepare_ego4d_subset_features.py `
    --video_dir datasets/ego4d/v2/full_scale `
    --output_dir outputs/ego4d_subset/features `
    --train_uid_file outputs/ego4d_subset/train_video_uids.txt `
    --val_uid_file outputs/ego4d_subset/val_video_uids.txt `
    --fps 2 `
    --resolution 384 `
    --batch_size 8 `
    --vision_pretrained google/siglip-large-patch16-384 `
    --device cuda
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
