$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HOME = "E:\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"
$python = "D:\Users\15373\anaconda3\envs\video-mm\python.exe"

& $python evaluate.py `
    --live_version live1+ `
    --llm_pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --eval_datasets ego4d_refined_narration_stream_val `
    --ego4d_val_video_uid_file outputs/ego4d_subset/val_video_uids.txt `
    --ego4d_feature_dir outputs/ego4d_subset/features `
    --ego4d_val_annotation_file outputs/ego4d_subset/refined_narration_stream_val_subset.json `
    --resume_from_checkpoint outputs/sanity/tinyllama_ego4d_subset `
    --per_device_eval_batch_size 1 `
    --prediction_loss_only False `
    --dataloader_num_workers 0 `
    --bf16 True `
    --tf32 False `
    --report_to none `
    --attn_implementation sdpa `
    --output_dir outputs/sanity/tinyllama_ego4d_subset_eval
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
