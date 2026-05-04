import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data import build_concat_train_dataset, get_data_collator
from models import build_model_and_tokenizer
from models.arguments_live import LiveOnePlusTrainingArguments


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/sanity/tinyllama_demo_overfit")
    parser.add_argument("--feature_dir", default="outputs/demo_overfit/features")
    parser.add_argument("--output", default="outputs/sanity/tinyllama_demo_overfit/pred_label_compare.json")
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    args = LiveOnePlusTrainingArguments(
        llm_pretrained="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        train_datasets=["demo_overfit"],
        demo_overfit_feature_dir=args_cli.feature_dir,
        output_dir=args_cli.checkpoint,
        resume_from_checkpoint=args_cli.checkpoint,
        attn_implementation="sdpa",
        bf16=True,
    )
    kwargs = asdict(args)

    model, tokenizer = build_model_and_tokenizer(is_training=False, set_vision_inside=False, **kwargs)
    model.to(args_cli.device).eval()
    dataset = build_concat_train_dataset(tokenizer=tokenizer, **kwargs)
    collator = get_data_collator(tokenizer=tokenizer, **kwargs)

    results = []
    correct = 0
    total = 0
    with torch.inference_mode():
        for sample_idx in range(len(dataset)):
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
            token_correct = sum(int(p == y) for p, y in zip(pred_ids, label_ids))
            correct += token_correct
            total += len(label_ids)
            results.append({
                "sample_idx": sample_idx,
                "token_accuracy": token_correct / max(1, len(label_ids)),
                "num_label_tokens": len(label_ids),
                "label_text": decode(tokenizer, label_ids),
                "pred_text": decode(tokenizer, pred_ids),
            })

    payload = {
        "checkpoint": args_cli.checkpoint,
        "feature_dir": args_cli.feature_dir,
        "overall_token_accuracy": correct / max(1, total),
        "num_supervised_tokens": total,
        "samples": results,
    }

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output}")
    print(f"overall_token_accuracy={payload['overall_token_accuracy']:.6f} supervised_tokens={total}")
    for result in results:
        print(f"\n=== sample {result['sample_idx']} token_acc={result['token_accuracy']:.6f} ===")
        print("[LABEL]")
        print(result["label_text"])
        print("[PRED]")
        print(result["pred_text"])


if __name__ == "__main__":
    main()
