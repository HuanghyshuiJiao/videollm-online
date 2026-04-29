$env:TOKENIZERS_PARALLELISM = "false"

python train.py `
    --live_version live1+ `
    --llm_pretrained TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --train_datasets robustness `
    --max_steps 20 `
    --per_device_train_batch_size 1 `
    --gradient_accumulation_steps 1 `
    --gradient_checkpointing True `
    --eval_strategy no `
    --prediction_loss_only True `
    --save_strategy no `
    --learning_rate 0.0002 `
    --optim adamw_torch `
    --lr_scheduler_type constant `
    --logging_steps 1 `
    --dataloader_num_workers 0 `
    --fp16 True `
    --bf16 False `
    --tf32 True `
    --report_to none `
    --attn_implementation sdpa `
    --lora_r 8 `
    --lora_alpha 16 `
    --output_dir outputs/sanity/tinyllama_robustness
