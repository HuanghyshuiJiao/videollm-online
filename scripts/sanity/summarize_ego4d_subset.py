import json
from pathlib import Path


def read_uids(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten_groups(groups: dict) -> list[dict]:
    narrations = [item for group in groups.values() for item in group]
    return sorted(narrations, key=lambda item: item.get("time", 0))


def main():
    train_uids = read_uids("outputs/ego4d_subset/train_video_uids.txt")
    val_uids = read_uids("outputs/ego4d_subset/val_video_uids.txt")
    annotations = {
        "train": json.load(open("datasets/ego4d/v2/annotations/refined_narration_stream_train.json", encoding="utf-8")),
        "val": json.load(open("datasets/ego4d/v2/annotations/refined_narration_stream_val.json", encoding="utf-8")),
    }

    manifest = {}
    for split, uids in (("train", train_uids), ("val", val_uids)):
        rows = []
        for uid in uids:
            narrations = flatten_groups(annotations[split][uid])
            rows.append({
                "split": split,
                "video_uid": uid,
                "video_file": f"datasets/ego4d/v2/full_scale/{uid}.mp4",
                "feature_file": f"outputs/ego4d_subset/features/{uid}.pt",
                "annotation_key": f"refined_narration_stream_{split}.json[{uid}]",
                "num_narrations": len(narrations),
                "time_start": narrations[0]["time"] if narrations else None,
                "time_end": narrations[-1]["time"] if narrations else None,
                "examples": narrations[:3],
            })
        manifest[split] = rows

    output = Path("outputs/ego4d_subset/manifest_readable.json")
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    for split, rows in manifest.items():
        print(f"{split}: {len(rows)} videos")
        for row in rows:
            print(
                f"{row['video_uid']}  narrations={row['num_narrations']}  "
                f"time={row['time_start']:.2f}-{row['time_end']:.2f}s"
            )


if __name__ == "__main__":
    main()
