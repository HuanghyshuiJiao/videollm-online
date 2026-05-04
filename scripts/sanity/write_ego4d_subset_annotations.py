import json
from pathlib import Path


def read_uids(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_subset(src: str, uid_file: str, dst: str):
    uids = read_uids(uid_file)
    data = json.load(open(src, encoding="utf-8"))
    subset = {uid: data[uid] for uid in uids if uid in data}
    output = Path(dst)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(subset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{src} -> {dst} ({len(subset)} videos)")


def main():
    write_subset(
        "datasets/ego4d/v2/annotations/refined_narration_stream_train.json",
        "outputs/ego4d_subset/train_video_uids.txt",
        "outputs/ego4d_subset/refined_narration_stream_train_subset.json",
    )
    write_subset(
        "datasets/ego4d/v2/annotations/refined_narration_stream_val.json",
        "outputs/ego4d_subset/val_video_uids.txt",
        "outputs/ego4d_subset/refined_narration_stream_val_subset.json",
    )


if __name__ == "__main__":
    main()
