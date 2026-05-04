import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
import transformers
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.arguments_live import LiveOnePlusTrainingArguments
from models.configuration_live import LiveConfigMixin
from models.vision_live import build_live_vision


def read_uids(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def resize_and_pad(frames: torch.Tensor, resolution: int) -> torch.Tensor:
    frames = frames.float()
    _, _, height, width = frames.shape
    if width >= height:
        new_width = resolution
        new_height = max(1, round(height * resolution / width))
    else:
        new_height = resolution
        new_width = max(1, round(width * resolution / height))
    frames = F.interpolate(frames, size=(new_height, new_width), mode="bicubic", align_corners=False)
    pad_left = (resolution - new_width) // 2
    pad_right = resolution - new_width - pad_left
    pad_top = (resolution - new_height) // 2
    pad_bottom = resolution - new_height - pad_top
    return F.pad(frames, (pad_left, pad_right, pad_top, pad_bottom))


def sample_video(path: str, fps: int, resolution: int) -> torch.Tensor:
    reader = torchvision.io.VideoReader(path, "video")
    frames = []
    next_time = 0.0
    for frame in reader:
        if frame["pts"] + 1e-6 < next_time:
            continue
        frames.append(frame["data"])
        next_time += 1 / fps
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return resize_and_pad(torch.stack(frames), resolution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="datasets/ego4d/v2/full_scale")
    parser.add_argument("--output_dir", default="datasets/ego4d/v2/full_scale_2fps_384_1+3x3_google--siglip-large-patch16-384")
    parser.add_argument("--train_uid_file", default="outputs/ego4d_subset/train_video_uids.txt")
    parser.add_argument("--val_uid_file", default="outputs/ego4d_subset/val_video_uids.txt")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vision_pretrained", default="google/siglip-large-patch16-384")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    uids = read_uids(args.train_uid_file) + read_uids(args.val_uid_file)
    os.makedirs(args.output_dir, exist_ok=True)

    live_args = LiveOnePlusTrainingArguments(vision_pretrained=args.vision_pretrained)
    config = LiveConfigMixin(**asdict(live_args))
    vision_model, vision_encode = build_live_vision(config)
    vision_model.to(args.device).eval()

    for uid in tqdm(uids, desc="ego4d subset features"):
        video_path = os.path.join(args.video_dir, f"{uid}.mp4")
        save_path = os.path.join(args.output_dir, f"{uid}.pt")
        if os.path.exists(save_path):
            continue
        frames = sample_video(video_path, args.fps, args.resolution)
        embeds = []
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=args.device.startswith("cuda")):
            for batch in frames.split(args.batch_size):
                embeds.append(vision_encode(vision_model, batch.to(args.device)).cpu())
        torch.save(torch.cat(embeds).to(torch.bfloat16), save_path)
        print(f"{video_path} -> {save_path}")


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    main()
