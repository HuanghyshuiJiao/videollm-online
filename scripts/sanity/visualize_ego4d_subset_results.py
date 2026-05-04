import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("outputs/ppt/ego4d_subset")
TRAINER_STATE = Path("outputs/sanity/tinyllama_ego4d_subset/checkpoint-200/trainer_state.json")
SELECTED_VIDEOS = Path("outputs/ego4d_subset/selected_videos.json")

VAL_METRICS = {
    "lm_ppl": 5.560096263885498,
    "time_diff_sec": 2.8853671550750732,
    "fluency": 0.09538281708955765,
    "lm_correctness": 0.40819546580314636,
    "val_samples": 13,
}


def save_loss_curve(log_history: list[dict]):
    rows = [row for row in log_history if "loss" in row]
    steps = [row["step"] for row in rows]
    losses = [row["loss"] for row in rows]

    plt.figure(figsize=(10, 5.6), dpi=180)
    plt.plot(steps, losses, color="#2563eb", linewidth=1.8, alpha=0.78, label="training loss")

    window = 10
    if len(losses) >= window:
        moving = [
            sum(losses[max(0, i - window + 1): i + 1]) / len(losses[max(0, i - window + 1): i + 1])
            for i in range(len(losses))
        ]
        plt.plot(steps, moving, color="#dc2626", linewidth=2.6, label="10-step moving avg")

    plt.title("TinyLlama LoRA on Ego4D Subset: Training Loss", fontsize=16, pad=14)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.28)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_loss_curve.png", bbox_inches="tight")
    plt.close()


def save_val_metrics():
    labels = ["LM PPL", "Time Diff (s)", "Fluency", "LM Correctness"]
    values = [
        VAL_METRICS["lm_ppl"],
        VAL_METRICS["time_diff_sec"],
        VAL_METRICS["fluency"],
        VAL_METRICS["lm_correctness"],
    ]
    colors = ["#2563eb", "#7c3aed", "#ea580c", "#16a34a"]

    plt.figure(figsize=(10, 5.6), dpi=180)
    bars = plt.bar(labels, values, color=colors, alpha=0.88)
    plt.title("Validation Metrics on 3 Held-Out Ego4D Videos", fontsize=16, pad=14)
    plt.ylabel("Metric value")
    plt.grid(axis="y", linestyle="--", alpha=0.24)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "validation_metrics.png", bbox_inches="tight")
    plt.close()


def save_dataset_summary(selected: dict):
    splits = ["train", "val"]
    video_counts = [len(selected[split]) for split in splits]
    narration_counts = [sum(item["num_narrations"] for item in selected[split]) for split in splits]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), dpi=180)
    axes[0].bar(["Train", "Val"], video_counts, color=["#2563eb", "#16a34a"], alpha=0.88)
    axes[0].set_title("Video Count")
    axes[0].set_ylabel("Videos")
    axes[0].grid(axis="y", linestyle="--", alpha=0.24)
    for i, value in enumerate(video_counts):
        axes[0].text(i, value, str(value), ha="center", va="bottom", fontsize=12)

    axes[1].bar(["Train", "Val"], narration_counts, color=["#2563eb", "#16a34a"], alpha=0.88)
    axes[1].set_title("Narration Count")
    axes[1].set_ylabel("Narrations")
    axes[1].grid(axis="y", linestyle="--", alpha=0.24)
    for i, value in enumerate(narration_counts):
        axes[1].text(i, value, str(value), ha="center", va="bottom", fontsize=12)

    fig.suptitle("Ego4D Mini-Experiment Dataset", fontsize=16)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "dataset_summary.png", bbox_inches="tight")
    plt.close(fig)


def save_pipeline_diagram():
    steps = [
        "13 Ego4D videos",
        "Refined narration subset",
        "2 FPS SigLIP features",
        "TinyLlama-1.1B LoRA",
        "Validation metrics",
    ]

    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=180)
    ax.set_axis_off()
    x_positions = [0.08, 0.29, 0.50, 0.71, 0.91]
    y = 0.54
    for i, (x, label) in enumerate(zip(x_positions, steps)):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=11,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#334155", linewidth=1.3),
        )
        if i < len(x_positions) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.075, y),
                xytext=(x + 0.075, y),
                arrowprops=dict(arrowstyle="->", color="#334155", lw=1.8),
            )
    ax.set_title("Ego4D Subset Training Pipeline", fontsize=16, pad=18)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "pipeline_diagram.png", bbox_inches="tight")
    plt.close(fig)


def save_summary(selected: dict, log_history: list[dict]):
    loss_rows = [row for row in log_history if "loss" in row]
    first_loss = loss_rows[0]["loss"]
    final_loss = loss_rows[-1]["loss"]
    last_20 = sum(row["loss"] for row in loss_rows[-20:]) / 20
    train_narrations = sum(item["num_narrations"] for item in selected["train"])
    val_narrations = sum(item["num_narrations"] for item in selected["val"])

    summary = {
        "dataset": {
            "train_videos": len(selected["train"]),
            "val_videos": len(selected["val"]),
            "train_narrations": train_narrations,
            "val_narrations": val_narrations,
        },
        "training": {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "method": "LoRA r=8 alpha=16",
            "steps": 200,
            "first_loss": first_loss,
            "final_loss": final_loss,
            "last_20_step_avg_loss": last_20,
        },
        "validation": VAL_METRICS,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = f"""# Ego4D Mini-Experiment Summary

## Setup
- Train videos: {len(selected["train"])} ({train_narrations} narrations)
- Val videos: {len(selected["val"])} ({val_narrations} narrations)
- Model: TinyLlama-1.1B Chat + LoRA (r=8, alpha=16)
- Steps: 200

## Training
- First logged loss: {first_loss:.4f}
- Final logged loss: {final_loss:.4f}
- Last 20-step average loss: {last_20:.4f}

## Validation
- LM PPL: {VAL_METRICS["lm_ppl"]:.4f}
- Time diff: {VAL_METRICS["time_diff_sec"]:.4f} s
- Fluency: {VAL_METRICS["fluency"]:.4f}
- LM correctness: {VAL_METRICS["lm_correctness"]:.4f}
"""
    (RESULTS_DIR / "summary.md").write_text(markdown, encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trainer_state = json.load(open(TRAINER_STATE, encoding="utf-8"))
    selected = json.load(open(SELECTED_VIDEOS, encoding="utf-8"))

    save_loss_curve(trainer_state["log_history"])
    save_val_metrics()
    save_dataset_summary(selected)
    save_pipeline_diagram()
    save_summary(selected, trainer_state["log_history"])
    print(f"Wrote PPT assets to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
