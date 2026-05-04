import argparse
import json
import random
from pathlib import Path


def load_existing_uids(video_dir: Path) -> set[str]:
    if not video_dir.exists():
        return set()
    return {path.stem for path in video_dir.glob("*.mp4")}


def load_annotation(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["videos"] if "videos" in data else data


def summarize_video(video_uid: str, anno: dict) -> dict:
    if "narrations" in anno:
        narrations = anno.get("narrations", [])
        duration = float(anno.get("duration_sec") or anno.get("duration") or 0)
    else:
        narrations = [item for items in anno.values() for item in items]
        duration = max((float(item.get("time", 0)) for item in narrations), default=0)
    return {
        "video_uid": video_uid,
        "num_narrations": len(narrations),
        "duration_sec": duration,
    }


def collect_candidates(annotation: dict, min_narrations: int, max_duration: float, existing_uids: set[str]) -> list[dict]:
    candidates = []
    for video_uid, anno in annotation.items():
        item = summarize_video(video_uid, anno)
        if item["num_narrations"] < min_narrations:
            continue
        if item["duration_sec"] and item["duration_sec"] > max_duration:
            continue
        item["already_downloaded"] = video_uid in existing_uids
        candidates.append(item)
    return candidates


def choose(candidates: list[dict], count: int, seed: int) -> list[dict]:
    if len(candidates) < count:
        raise RuntimeError(f"Only found {len(candidates)} candidates; need {count}.")
    random.seed(seed)
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (not item["already_downloaded"], item["duration_sec"] or 10**9))
    return candidates[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", default="datasets/ego4d/v2/annotations/all_narrations_redacted.json")
    parser.add_argument("--train_annotation", default=None)
    parser.add_argument("--val_annotation", default=None)
    parser.add_argument("--output_dir", default="outputs/ego4d_subset")
    parser.add_argument("--train_count", type=int, default=10)
    parser.add_argument("--val_count", type=int, default=3)
    parser.add_argument("--min_narrations", type=int, default=20)
    parser.add_argument("--max_duration", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefer_existing_video_dir", default=None)
    args = parser.parse_args()

    train_annotation = load_annotation(args.train_annotation or args.annotation)
    val_annotation = load_annotation(args.val_annotation or args.annotation)
    existing_uids = load_existing_uids(Path(args.prefer_existing_video_dir)) if args.prefer_existing_video_dir else set()

    train_candidates = collect_candidates(train_annotation, args.min_narrations, args.max_duration, existing_uids)
    val_candidates = collect_candidates(val_annotation, args.min_narrations, args.max_duration, existing_uids)
    train = choose(train_candidates, args.train_count, args.seed)
    train_uids = {item["video_uid"] for item in train}
    val = choose([item for item in val_candidates if item["video_uid"] not in train_uids], args.val_count, args.seed + 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_video_uids.txt").write_text(
        "\n".join(item["video_uid"] for item in train) + "\n",
        encoding="utf-8",
    )
    (output_dir / "val_video_uids.txt").write_text(
        "\n".join(item["video_uid"] for item in val) + "\n",
        encoding="utf-8",
    )
    (output_dir / "selected_videos.json").write_text(
        json.dumps({"train": train, "val": val}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'train_video_uids.txt'}")
    print(f"Wrote {output_dir / 'val_video_uids.txt'}")
    print(f"Wrote {output_dir / 'selected_videos.json'}")


if __name__ == "__main__":
    main()
