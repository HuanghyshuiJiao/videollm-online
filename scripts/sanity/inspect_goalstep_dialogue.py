import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="datasets/ego4d/v2/annotations/goalstep_livechat_trainval_filtered_21k.json",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--stats-limit", type=int, default=1000)
    args = parser.parse_args()

    path = Path(args.path)
    data = json.load(open(path, encoding="utf-8"))
    sample = data[args.index]
    conversation = sample.get("conversation", [])

    print(f"path: {path}")
    print(f"num_items: {len(data)}")
    print(f"index: {args.index}")
    print(f"video_uid: {sample.get('video_uid')}")
    print(f"duration: {sample.get('duration')}")
    print(f"top_keys: {list(sample.keys())}")
    print(f"conversation_len: {len(conversation)}")

    times = [message["time"] for message in conversation if "time" in message]
    if times:
        print(f"conversation_time_range: {min(times)} -> {max(times)}")

    print("\nmessages:")
    for i, message in enumerate(conversation[: args.head]):
        content = message.get("content", "").replace("\n", " ")
        if len(content) > 180:
            content = content[:177] + "..."
        print(
            f"[{i:02d}] role={message.get('role')} "
            f"time={message.get('time')} content={content}"
        )

    role_counts = collections.Counter()
    lengths = []
    all_times = []
    for item in data[: args.stats_limit]:
        item_conversation = item.get("conversation", [])
        lengths.append(len(item_conversation))
        for message in item_conversation:
            role_counts[message.get("role")] += 1
            if "time" in message:
                all_times.append(message["time"])

    print(f"\nstats_first_{min(args.stats_limit, len(data))}:")
    print(f"role_counts: {dict(role_counts)}")
    print(
        "conversation_len_min_avg_max: "
        f"{min(lengths)} / {sum(lengths) / len(lengths):.2f} / {max(lengths)}"
    )
    if all_times:
        print(f"time_min_max: {min(all_times)} / {max(all_times)}")


if __name__ == "__main__":
    main()
