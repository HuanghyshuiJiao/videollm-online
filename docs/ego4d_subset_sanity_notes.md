# Ego4D Subset Sanity Experiment Notes

Last updated: 2026-05-04

This note records the small Ego4D training/validation workflow we built and ran, so a future Codex session can quickly recover context.

## Goal

Run a small real-data sanity experiment after Ego4D access was granted:

1. Download a small number of Ego4D videos.
2. Match them with available narration training data.
3. Extract SigLIP video features.
4. Fine-tune a small LLM (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) with LoRA.
5. Evaluate on a tiny validation subset.
6. Produce PPT-friendly visualizations.

This followed an earlier demo-video overfit sanity test.

## Branch And Commit

Repo branch:

```text
small-llm-sanity
```

Pushed commit:

```text
0e71a32 Add Ego4D subset sanity training workflow
```

Remote:

```text
origin https://github.com/HuanghyshuiJiao/videollm-online.git
```

## Important Safety Notes

- Real Ego4D/AWS credentials must never be committed.
- The local credentials file was deleted after use:

```text
outputs/ego4d_credentials.ini
```

- The committed credential file is only a placeholder template:

```text
scripts/sanity/ego4d_credentials.template.ini
```

- Large/local outputs are ignored by `.gitignore`:

```text
/outputs
/datasets
/checkpoints
/ffmpeg
```

## Storage Layout

We moved physical Ego4D storage to E drive and kept the repo path stable by using a Windows junction:

```text
videollm-online/datasets/ego4d -> E:\datasets\ego4d
```

The code still reads:

```text
datasets/ego4d/v2
```

Physical video location:

```text
E:\datasets\ego4d\v2\full_scale
```

Original D-drive Ego4D folder was backed up locally as:

```text
datasets/ego4d_d_drive_backup_20260504_011803
```

## Existing Local Model Cache

SigLIP was already present locally and was reused:

```text
E:\hf-cache\hub\models--google--siglip-large-patch16-384
```

Scripts set:

```powershell
$env:HF_HOME = "E:\hf-cache"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

## Dataset Subset

We used refined Ego4D narration annotations already present locally:

```text
datasets/ego4d/v2/annotations/refined_narration_stream_train.json
datasets/ego4d/v2/annotations/refined_narration_stream_val.json
```

We selected:

```text
10 train videos
3 val videos
```

UID lists:

```text
outputs/ego4d_subset/train_video_uids.txt
outputs/ego4d_subset/val_video_uids.txt
```

Readable manifest:

```text
outputs/ego4d_subset/manifest_readable.json
```

Selected-video summary:

```text
outputs/ego4d_subset/selected_videos.json
```

Subset annotation files created to avoid loading the full 746MB refined train JSON during model training:

```text
outputs/ego4d_subset/refined_narration_stream_train_subset.json
outputs/ego4d_subset/refined_narration_stream_val_subset.json
```

Training dataset size after conversion:

```text
train_len = 44 stream samples
val_len = 13 stream samples
```

Narration counts:

```text
train: 248 narrations
val: 103 narrations
```

## Downloaded Videos

Downloaded 13 full-scale Ego4D videos to:

```text
E:\datasets\ego4d\v2\full_scale
```

The download script:

```text
scripts/sanity/download_ego4d_subset.ps1
```

## Feature Extraction

Because upstream preprocessing uses `submitit`, which fails on Windows local mode due to Unix-only signals (`SIGCONT`, `SIGKILL`), we added a direct Windows-friendly feature extractor:

```text
scripts/sanity/prepare_ego4d_subset_features.py
scripts/sanity/preprocess_ego4d_subset.ps1
```

It:

1. Reads the selected MP4 files.
2. Samples at 2 FPS.
3. Resizes/pads frames to 384.
4. Encodes with `google/siglip-large-patch16-384`.
5. Saves bf16 `.pt` features.

Feature output:

```text
outputs/ego4d_subset/features
```

13 `.pt` files were generated.

Metadata cache:

```text
outputs/ego4d_subset/features_metadata.json
```

## Code Changes

### `models/arguments_live.py`

Added arguments:

```python
demo_overfit_feature_dir
ego4d_train_video_uid_file
ego4d_val_video_uid_file
ego4d_feature_dir
ego4d_train_annotation_file
ego4d_val_annotation_file
```

### `data/ego4d/ego4d.py`

Added support for overriding the feature directory:

```python
ego4d_feature_dir
```

This lets tiny experiments store features in:

```text
outputs/ego4d_subset/features
```

instead of the full default Ego4D feature directory.

### `data/ego4d/narration.py`

Added support for:

- UID subset filtering.
- Missing feature skipping.
- Refined annotation subset files.

This allows `ego4d_refined_narration_stream_train` and `ego4d_refined_narration_stream_val` to run with tiny local JSON files.

### `data/data_collator.py`

Fixed a tensor indexing issue:

```python
torch.nonzero(...).item()
```

became:

```python
torch.nonzero(...).flatten()[0].item()
```

This avoids errors when `torch.nonzero` returns a non-scalar tensor.

### `data/demo_overfit.py`

Added a tiny two-video demo overfit dataset used by earlier sanity tests.

## Important Scripts

### Credential Setup

Template:

```text
scripts/sanity/ego4d_credentials.template.ini
```

Install filled credentials into AWS config:

```powershell
.\scripts\sanity\setup_ego4d_credentials.ps1 -Force
```

### Storage Setup

Create E-drive storage and repo junction:

```powershell
.\scripts\sanity\setup_ego4d_storage.ps1 -Ego4DRoot E:\datasets\ego4d
```

### Select Ego4D Subset

```powershell
.\scripts\sanity\prepare_ego4d_subset.ps1
```

Underlying Python:

```text
scripts/sanity/select_ego4d_narration_subset.py
```

### Download Videos

```powershell
.\scripts\sanity\download_ego4d_subset.ps1
```

### Extract Features

```powershell
.\scripts\sanity\preprocess_ego4d_subset.ps1
```

### Write Small Annotation Files

```powershell
D:\Users\15373\anaconda3\envs\video-mm\python.exe .\scripts\sanity\write_ego4d_subset_annotations.py
```

### Check Dataset Without Loading Full Training Model

```powershell
.\scripts\sanity\check_ego4d_subset_dataset.ps1
```

Observed output:

```text
train_len=44
ego4d_refined_narration_stream_val_len=13
sample_text_chars=1727
sample_frames_shape=(33, 10, 1024)
sample_learn_ranges=44
```

### Train TinyLlama On Ego4D Subset

```powershell
.\scripts\sanity\tinyllama_ego4d_subset.ps1
```

Training settings:

```text
Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
LoRA r: 8
LoRA alpha: 16
max_steps: 200
batch size: 1
gradient accumulation: 1
bf16: True
attention: sdpa
eval_strategy: no
```

We disabled mid-training eval because the current Transformers version hit:

```text
TypeError: 'dict' object is not callable
```

when `Trainer` evaluated dict eval datasets with dict compute metrics. Eval is run separately.

### Evaluate

```powershell
.\scripts\sanity\eval_tinyllama_ego4d_subset.ps1
```

### Visualize For PPT

```powershell
D:\Users\15373\anaconda3\envs\video-mm\python.exe .\scripts\sanity\visualize_ego4d_subset_results.py
```

Outputs:

```text
outputs/ppt/ego4d_subset/training_loss_curve.png
outputs/ppt/ego4d_subset/validation_metrics.png
outputs/ppt/ego4d_subset/dataset_summary.png
outputs/ppt/ego4d_subset/pipeline_diagram.png
outputs/ppt/ego4d_subset/summary.md
outputs/ppt/ego4d_subset/summary.json
```

## Training Output

Checkpoint:

```text
outputs/sanity/tinyllama_ego4d_subset
```

Contains:

```text
checkpoint-100
checkpoint-200
adapter_model.safetensors
adapter_config.json
tokenizer.json
training_args.bin
```

Training completed successfully with exit code 0.

Training loss:

```text
first logged loss: 8.7063
final logged loss: 0.2026
last 20-step average loss: 0.2603
```

## Validation Results

Eval-only command completed successfully.

Metrics:

```text
LM PPL:         5.5601
Time diff:      2.8854 seconds
Fluency:        0.0954
LM correctness: 0.4082
Val samples:    13
```

Interpretation:

- The pipeline works end to end.
- The tiny model strongly fits the small train subset.
- Validation is weak, as expected for 10 train videos and 3 val videos.
- `fluency` and `time_diff` suggest the online timing behavior is still poor at this scale.

## PPT Summary Numbers

```text
Train: 10 videos / 248 narrations
Val: 3 videos / 103 narrations
Model: TinyLlama-1.1B Chat + LoRA r=8 alpha=16
Steps: 200
Loss: 8.7063 -> 0.2026
Val LM PPL: 5.5601
Val time diff: 2.8854s
Val fluency: 0.0954
Val LM correctness: 0.4082
```

## Known Issues / Gotchas

1. `submitit` local mode fails on Windows because it expects Unix signals.
   Use `scripts/sanity/prepare_ego4d_subset_features.py` for small local runs.

2. Loading full refined train annotation after model load can trigger memory issues.
   Use subset annotation files:

   ```text
   outputs/ego4d_subset/refined_narration_stream_train_subset.json
   outputs/ego4d_subset/refined_narration_stream_val_subset.json
   ```

3. Mid-training eval with dict eval datasets hit `TypeError: 'dict' object is not callable`.
   Current workaround: train with `--eval_strategy no`, then run `eval_tinyllama_ego4d_subset.ps1`.

4. Some scripts contain machine-specific paths:

   ```text
   D:\Users\15373\anaconda3\envs\video-mm\python.exe
   E:\hf-cache
   E:\datasets\ego4d
   ```

5. `outputs/` and `datasets/` are ignored and not pushed. The new conversation should not expect local data to exist unless using the same machine.

## Suggested Next Steps

1. Increase data scale:

   ```text
   train: 50-100 videos
   val: 10-20 videos
   ```

2. Increase training:

   ```text
   max_steps: 500-1000
   ```

3. Keep the same pipeline:

   ```text
   select subset -> download videos -> extract features -> write subset annotations -> check dataset -> train -> eval -> visualize
   ```

4. Compare new validation metrics against this baseline:

   ```text
   LM PPL 5.5601
   Time diff 2.8854s
   Fluency 0.0954
   LM correctness 0.4082
   ```

