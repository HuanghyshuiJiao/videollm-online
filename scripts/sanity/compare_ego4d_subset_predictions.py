import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import build_eval_dataset_dict, get_data_collator
from models import build_model_and_tokenizer
from models.arguments_live import LiveOnePlusTrainingArguments


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/sanity/tinyllama_ego4d_subset")
    parser.add_argument("--feature_dir", default="outputs/ego4d_subset/features")
    parser.add_argument("--val_uid_file", default="outputs/ego4d_subset/val_video_uids.txt")
    parser.add_argument("--val_annotation_file", default="outputs/ego4d_subset/refined_narration_stream_val_subset.json")
    parser.add_argument("--output", default="outputs/ppt/ego4d_subset/pred_gt_examples.json")
    parser.add_argument("--markdown", default="outputs/ppt/ego4d_subset/pred_gt_examples.md")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    args = LiveOnePlusTrainingArguments(
        llm_pretrained="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        eval_datasets=["ego4d_refined_narration_stream_val"],
        ego4d_val_video_uid_file=args_cli.val_uid_file,
        ego4d_feature_dir=args_cli.feature_dir,
        ego4d_val_annotation_file=args_cli.val_annotation_file,
        output_dir=args_cli.checkpoint,
        resume_from_checkpoint=args_cli.checkpoint,
        attn_implementation="sdpa",
        bf16=True,
    )
    kwargs = asdict(args)

    model, tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=False, **kwargs)
    model.to(args_cli.device).eval()
    dataset = build_eval_dataset_dict(tokenizer=tokenizer, **kwargs)["ego4d_refined_narration_stream_val"]
    collator = get_data_collator(tokenizer=tokenizer, **kwargs)

    results = []
    correct = 0
    total = 0
    with torch.inference_mode():
        for sample_idx in range(min(args_cli.num_samples, len(dataset))):
            batch = collator([dataset[sample_idx]])
            batch = {key: value.to(args_cli.device) if hasattr(value, "to") else value for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                frames=batch["frames"],
                labels=batch["labels"],
                return_dict=True,
            )
            mask = batch["labels"][0] >= 0
            label_ids = batch["labels"][0][mask].detach().cpu().tolist()
            pred_ids = outputs.logits[0].argmax(dim=-1)[mask].detach().cpu().tolist()
            token_correct = sum(int(pred == label) for pred, label in zip(pred_ids, label_ids))
            correct += token_correct
            total += len(label_ids)
            results.append({
                "sample_idx": sample_idx,
                "token_accuracy": token_correct / max(1, len(label_ids)),
                "num_label_tokens": len(label_ids),
                "gt_text": decode(tokenizer, label_ids),
                "pred_text": decode(tokenizer, pred_ids),
            })

    payload = {
        "note": "Teacher-forced argmax predictions on supervised label positions, not free-form generation.",
        "checkpoint": args_cli.checkpoint,
        "overall_token_accuracy_on_dumped_samples": correct / max(1, total),
        "num_dumped_samples": len(results),
        "samples": results,
    }

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Ego4D Val Prediction vs GT Examples",
        "",
        "Teacher-forced argmax predictions on supervised label positions; this is for qualitative inspection, not free-form generation.",
        "",
        f"Overall token accuracy on dumped samples: {payload['overall_token_accuracy_on_dumped_samples']:.4f}",
        "",
    ]
    for item in results:
        lines.extend([
            f"## Sample {item['sample_idx']}",
            "",
            f"Token accuracy: {item['token_accuracy']:.4f} ({item['num_label_tokens']} supervised tokens)",
            "",
            "**GT**",
            "",
            "```text",
            item["gt_text"],
            "```",
            "",
            "**Pred**",
            "",
            "```text",
            item["pred_text"],
            "```",
            "",
        ])
    Path(args_cli.markdown).write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved {output}")
    print(f"Saved {args_cli.markdown}")
    print(f"overall_token_accuracy_on_dumped_samples={payload['overall_token_accuracy_on_dumped_samples']:.6f}")


if __name__ == "__main__":
    main()
