$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HOME = "E:\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$python = "D:\Users\15373\anaconda3\envs\video-mm\python.exe"

& $python .\scripts\sanity\prepare_demo_overfit.py `
    --video_dir demo/assets `
    --output_dir outputs/demo_overfit/features `
    --feature_mode siglip `
    --batch_size 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python train.py `
    --live_version live1+ `
    --llm_pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --train_datasets demo_overfit `
    --demo_overfit_feature_dir outputs/demo_overfit/features `
    --max_steps 80 `
    --per_device_train_batch_size 1 `
    --gradient_accumulation_steps 1 `
    --gradient_checkpointing True `
    --eval_strategy no `
    --prediction_loss_only True `
    --save_strategy steps `
    --save_steps 80 `
    --learning_rate 0.0005 `
    --optim adamw_torch `
    --lr_scheduler_type constant `
    --logging_steps 1 `
    --dataloader_num_workers 0 `
    --fp16 False `
    --bf16 True `
    --tf32 False `
    --report_to none `
    --attn_implementation sdpa `
    --lora_r 8 `
    --lora_alpha 16 `
    --output_dir outputs/sanity/tinyllama_demo_overfit
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
