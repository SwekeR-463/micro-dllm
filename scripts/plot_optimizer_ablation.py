import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate run logs and plot optimizer ablation curves."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("artifacts/runs"),
        help="Root directory containing run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/media/optimizer_ablation"),
        help="Output directory for plots and summary.",
    )
    parser.add_argument(
        "--ema-beta",
        type=float,
        default=0.98,
        help="EMA smoothing factor for noisy per-step curves.",
    )
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ema(values, beta):
    if not values:
        return []
    out = []
    acc = values[0]
    for v in values:
        acc = beta * acc + (1.0 - beta) * v
        out.append(acc)
    return out


def normalize_optimizer_name(name):
    low = name.lower()
    if "muon" in low:
        return "muon+adamw"
    if "adamw" in low:
        return "adamw"
    if low == "adam":
        return "adam"
    if "sgd" in low:
        return "sgd"
    return low


def load_runs(runs_root):
    runs = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        final_path = run_dir / "final_metrics.json"
        train_path = run_dir / "train_log.jsonl"
        eval_path = run_dir / "eval_log.jsonl"
        config_path = run_dir / "config.json"
        if not (final_path.exists() and train_path.exists() and eval_path.exists()):
            continue

        final_metrics = json.loads(final_path.read_text(encoding="utf-8"))
        train_rows = read_jsonl(train_path)
        eval_rows = read_jsonl(eval_path)
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        if not train_rows or not eval_rows:
            continue

        raw_opt = final_metrics.get("optimizer") or config.get("optimizer") or "unknown"
        opt = normalize_optimizer_name(raw_opt)
        runs.append(
            {
                "run_dir": run_dir,
                "run_id": final_metrics.get("run_id", run_dir.name),
                "optimizer": opt,
                "raw_optimizer": raw_opt,
                "train": train_rows,
                "eval": eval_rows,
                "final": final_metrics,
            }
        )
    return runs


def style_for_optimizers(optimizers):
    palette = {
        "adamw": "#1f77b4",
        "adam": "#ff7f0e",
        "sgd": "#2ca02c",
        "muon+adamw": "#d62728",
    }
    fallback = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    styles = {}
    idx = 0
    for opt in optimizers:
        if opt in palette:
            styles[opt] = palette[opt]
        else:
            styles[opt] = fallback[idx % len(fallback)]
            idx += 1
    return styles


def plot_eval_val_loss_vs_step(runs, color_map, out_path):
    plt.figure(figsize=(9, 5.5))
    for run in runs:
        x = [r["step"] for r in run["eval"]]
        y = [r["val_masked_loss"] for r in run["eval"]]
        plt.plot(x, y, marker="o", linewidth=2, markersize=4, label=run["optimizer"], color=color_map[run["optimizer"]])
    plt.title("Validation Masked Loss vs Step")
    plt.xlabel("Step")
    plt.ylabel("Val Masked Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_eval_val_loss_vs_time(runs, color_map, out_path):
    plt.figure(figsize=(9, 5.5))
    for run in runs:
        x = [r["time_s"] / 60.0 for r in run["eval"]]
        y = [r["val_masked_loss"] for r in run["eval"]]
        plt.plot(x, y, marker="o", linewidth=2, markersize=4, label=run["optimizer"], color=color_map[run["optimizer"]])
    plt.title("Validation Masked Loss vs Wall Time")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Val Masked Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_train_loss_ema_vs_step(runs, color_map, out_path, beta):
    plt.figure(figsize=(9, 5.5))
    for run in runs:
        x = [r["step"] for r in run["train"]]
        y = [r["loss"] for r in run["train"]]
        y_ema = ema(y, beta)
        plt.plot(x, y_ema, linewidth=2, label=run["optimizer"], color=color_map[run["optimizer"]])
    plt.title("Train Loss vs Step")
    plt.xlabel("Step")
    plt.ylabel("Train Loss (EMA)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_grad_norm_ema_vs_step(runs, color_map, out_path, beta):
    plt.figure(figsize=(9, 5.5))
    for run in runs:
        x = [r["step"] for r in run["train"]]
        y = [r["grad_norm"] for r in run["train"]]
        y_ema = ema(y, beta)
        plt.plot(x, y_ema, linewidth=2, label=run["optimizer"], color=color_map[run["optimizer"]])
    plt.title("Gradient Norm vs Step")
    plt.xlabel("Step")
    plt.ylabel("Grad Norm (EMA)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_perplexity_ema_vs_step(runs, color_map, out_path, beta):
    plt.figure(figsize=(9, 5.5))
    for run in runs:
        x = [r["step"] for r in run["train"]]
        y = [r["perplexity"] for r in run["train"]]
        y_ema = ema(y, beta)
        plt.plot(x, y_ema, linewidth=2, label=run["optimizer"], color=color_map[run["optimizer"]])
    plt.yscale("log")
    plt.title("Train Perplexity vs Step (log scale)")
    plt.xlabel("Step")
    plt.ylabel("Train Perplexity (EMA, log)")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_final_val_loss_bar(runs, color_map, out_path):
    runs_sorted = sorted(
        runs,
        key=lambda r: r["final"].get("final_val_masked_loss", math.inf),
    )
    labels = [r["optimizer"] for r in runs_sorted]
    vals = [r["final"].get("final_val_masked_loss", math.nan) for r in runs_sorted]
    colors = [color_map[r["optimizer"]] for r in runs_sorted]
    plt.figure(figsize=(9, 5.5))
    bars = plt.bar(labels, vals, color=colors)
    plt.title("Final Validation Masked Loss by Optimizer")
    plt.xlabel("Optimizer")
    plt.ylabel("Final Val Masked Loss")
    plt.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def write_summary(runs, out_path):
    rows = sorted(
        runs,
        key=lambda r: r["final"].get("final_val_masked_loss", math.inf),
    )
    lines = [
        "# Optimizer Ablation Summary",
        "",
        "| Rank | Optimizer | Final Val Loss | Final Val PPL | Final Recon Acc | Total Time (min) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for i, run in enumerate(rows, start=1):
        f = run["final"]
        time_min = f.get("total_time_s", 0.0) / 60.0
        lines.append(
            f"| {i} | {run['optimizer']} | {f.get('final_val_masked_loss', math.nan):.4f} | "
            f"{f.get('final_val_perplexity', math.nan):.2f} | {f.get('final_val_masked_recon_acc', math.nan):.4f} | "
            f"{time_min:.1f} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    runs = load_runs(args.runs_root)
    if not runs:
        raise SystemExit(f"No valid runs found under {args.runs_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    color_map = style_for_optimizers(sorted({r["optimizer"] for r in runs}))

    plot_eval_val_loss_vs_step(
        runs, color_map, args.out_dir / "01_val_masked_loss_vs_step.png"
    )
    plot_eval_val_loss_vs_time(
        runs, color_map, args.out_dir / "02_val_masked_loss_vs_time.png"
    )
    plot_train_loss_ema_vs_step(
        runs, color_map, args.out_dir / "03_train_loss_ema_vs_step.png", args.ema_beta
    )
    plot_perplexity_ema_vs_step(
        runs, color_map, args.out_dir / "04_train_perplexity_ema_vs_step.png", args.ema_beta
    )
    plot_grad_norm_ema_vs_step(
        runs, color_map, args.out_dir / "05_grad_norm_ema_vs_step.png", args.ema_beta
    )
    plot_final_val_loss_bar(
        runs, color_map, args.out_dir / "06_final_val_loss_bar.png"
    )
    write_summary(runs, args.out_dir / "summary.md")

    print(f"Saved plots and summary to: {args.out_dir}")


if __name__ == "__main__":
    main()
