import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_state", default="outputs/sanity/tinyllama_demo_overfit/checkpoint-80/trainer_state.json")
    parser.add_argument("--output", default="outputs/sanity/tinyllama_demo_overfit/loss_curve.png")
    args = parser.parse_args()

    state = json.loads(Path(args.trainer_state).read_text(encoding="utf-8"))
    points = [(entry["step"], entry["loss"]) for entry in state["log_history"] if "loss" in entry]
    if not points:
        raise RuntimeError(f"No loss entries found in {args.trainer_state}")

    steps, losses = zip(*points)
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, marker="o", markersize=3, linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Training loss (log scale)")
    plt.title("Demo Overfit Loss")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=160)
    print(f"Saved {output}")
    print(f"first_loss={losses[0]:.6g} last_loss={losses[-1]:.6g} min_loss={min(losses):.6g}")


if __name__ == "__main__":
    main()
