$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HOME = "E:\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"
$python = "D:\Users\15373\anaconda3\envs\video-mm\python.exe"

& $python train.py `
    --live_version live1+ `
    --llm_pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --train_datasets ego4d_refined_narration_stream_train `
    --eval_datasets ego4d_refined_narration_stream_val `
    --ego4d_train_video_uid_file outputs/ego4d_subset/train_video_uids.txt `
    --ego4d_val_video_uid_file outputs/ego4d_subset/val_video_uids.txt `
    --ego4d_feature_dir outputs/ego4d_subset/features `
    --ego4d_train_annotation_file outputs/ego4d_subset/refined_narration_stream_train_subset.json `
    --ego4d_val_annotation_file outputs/ego4d_subset/refined_narration_stream_val_subset.json `
    --max_steps 200 `
    --per_device_train_batch_size 1 `
    --per_device_eval_batch_size 1 `
    --gradient_accumulation_steps 1 `
    --gradient_checkpointing True `
    --eval_strategy no `
    --prediction_loss_only False `
    --save_strategy steps `
    --save_steps 100 `
    --learning_rate 0.0002 `
    --optim adamw_torch `
    --lr_scheduler_type cosine `
    --warmup_ratio 0.05 `
    --logging_steps 1 `
    --dataloader_num_workers 0 `
    --bf16 True `
    --tf32 False `
    --report_to none `
    --attn_implementation sdpa `
    --lora_r 8 `
    --lora_alpha 16 `
    --output_dir outputs/sanity/tinyllama_ego4d_subset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
