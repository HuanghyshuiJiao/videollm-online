$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HOME = "E:\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"
$python = "D:\Users\15373\anaconda3\envs\video-mm\python.exe"

& $python .\scripts\sanity\check_ego4d_subset_dataset.py `
    --live_version live1+ `
    --llm_pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --train_datasets ego4d_refined_narration_stream_train `
    --eval_datasets ego4d_refined_narration_stream_val `
    --ego4d_train_video_uid_file outputs/ego4d_subset/train_video_uids.txt `
    --ego4d_val_video_uid_file outputs/ego4d_subset/val_video_uids.txt `
    --ego4d_feature_dir outputs/ego4d_subset/features `
    --ego4d_train_annotation_file outputs/ego4d_subset/refined_narration_stream_train_subset.json `
    --ego4d_val_annotation_file outputs/ego4d_subset/refined_narration_stream_val_subset.json `
    --per_device_train_batch_size 1 `
    --per_device_eval_batch_size 1 `
    --dataloader_num_workers 0 `
    --attn_implementation sdpa `
    --output_dir outputs/sanity/check_ego4d_subset_dataset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
