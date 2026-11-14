"""Evaluate the saved Phase 1 model on the official UCI HAR test set.

    python phase1_model/evaluate.py

Loads phase1_model/weights.npz and prints test accuracy, per-class accuracy,
the confusion matrix, the parameter count, multiply-accumulate operations
(MACs) per window layer by layer, and batch-1 CPU latency. Writes
results.json and, if matplotlib is installed, confusion_matrix.png.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import har_data
from model import IN_CHANNELS, WINDOW, count_parameters, load_npz, standardise

HERE = Path(__file__).resolve().parent


def layer_table(model):
    """Per-layer output shape, parameters and MACs for ONE input window.

    One MAC = one multiply feeding an accumulator.
      Conv1d: out_len * out_channels * in_channels * kernel
      Linear: in_features * out_features
    Bias additions, ReLU and max-pool comparisons contain no multiplies and
    are counted as 0 MACs (max-pool comparisons are listed separately).
    """
    rows = []

    def hook(name):
        def fn(mod, inp, out):
            params = sum(p.numel() for p in mod.parameters())
            macs = compares = 0
            if isinstance(mod, nn.Conv1d):
                macs = out.shape[2] * mod.out_channels * mod.in_channels * mod.kernel_size[0]
            elif isinstance(mod, nn.Linear):
                macs = mod.in_features * mod.out_features
            elif isinstance(mod, nn.MaxPool1d):
                compares = out[0].numel() * (mod.kernel_size - 1)
            rows.append({"layer": name, "type": type(mod).__name__,
                         "input_shape": list(inp[0].shape[1:]),
                         "output_shape": list(out.shape[1:]),
                         "parameters": params, "macs": int(macs),
                         "compares": int(compares)})
        return fn

    # forward() order: conv1, relu, pool, conv2, relu, pool, fc
    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_children()]
    with torch.no_grad():
        model(torch.zeros(1, IN_CHANNELS, WINDOW))
    for h in handles:
        h.remove()
    return rows


def cpu_name():
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or platform.machine()


def measure_latency(model, x_raw, mean, std, threads, warmup=500, runs=5000):
    """Batch-size-1 latency: standardise one raw window + forward pass.

    Each timed call uses a different test window (cycling through the test
    set). Times are wall-clock from time.perf_counter_ns.
    """
    torch.set_num_threads(threads)
    mean_t = torch.from_numpy(mean)[None, :, None]
    std_t = torch.from_numpy(std)[None, :, None]
    windows = torch.from_numpy(x_raw)
    n = len(windows)
    times = np.empty(runs, dtype=np.int64)
    with torch.inference_mode():
        for i in range(warmup):
            model((windows[i % n:i % n + 1] - mean_t) / std_t)
        for i in range(runs):
            w = windows[i % n:i % n + 1]
            t0 = time.perf_counter_ns()
            model((w - mean_t) / std_t)
            times[i] = time.perf_counter_ns() - t0
    us = times / 1e3
    return {"threads": threads, "runs": runs, "warmup": warmup,
            "median_us": float(np.median(us)), "mean_us": float(us.mean()),
            "p99_us": float(np.percentile(us, 99)), "min_us": float(us.min())}


def save_confusion_png(cm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping confusion_matrix.png")
        return False
    labels = [c.replace("WALKING_", "WALK_").title() for c in har_data.CLASSES]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"UCI HAR test set, accuracy {np.trace(cm) / cm.sum() * 100:.2f}%")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=HERE / "weights.npz")
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args(argv)

    data = har_data.load()
    model, mean, std = load_npz(args.weights)
    x_raw, y = data["x_test"], data["y_test"]

    with torch.no_grad():
        pred = model(torch.from_numpy(standardise(x_raw, mean, std))).argmax(1).numpy()

    k = len(har_data.CLASSES)
    cm = np.zeros((k, k), dtype=np.int64)       # rows: true, columns: predicted
    np.add.at(cm, (y, pred), 1)
    accuracy = float(np.trace(cm) / cm.sum())
    per_class = cm.diagonal() / cm.sum(axis=1)

    print(f"Weights: {args.weights}")
    print(f"Test windows: {len(y)}")
    print(f"Test accuracy: {accuracy * 100:.2f}% ({np.trace(cm)}/{cm.sum()})\n")
    print("Per-class accuracy (recall):")
    for name, acc, n in zip(har_data.CLASSES, per_class, cm.sum(axis=1)):
        print(f"  {name:<20s} {acc * 100:6.2f}%  (n={n})")

    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(" " * 20 + "".join(f"{i:>6d}" for i in range(k)))
    for i, name in enumerate(har_data.CLASSES):
        print(f"{i} {name:<18s}" + "".join(f"{v:>6d}" for v in cm[i]))

    rows = layer_table(model)
    total_params = count_parameters(model)
    total_macs = sum(r["macs"] for r in rows)
    assert total_params == sum(r["parameters"] for r in rows)
    print("\nPer-layer cost for one window:")
    print(f"  {'layer':<8s}{'type':<11s}{'input':<12s}{'output':<12s}"
          f"{'params':>8s}{'MACs':>9s}{'compares':>10s}")
    for r in rows:
        print(f"  {r['layer']:<8s}{r['type']:<11s}{str(tuple(r['input_shape'])):<12s}"
              f"{str(tuple(r['output_shape'])):<12s}{r['parameters']:>8d}"
              f"{r['macs']:>9d}{r['compares']:>10d}")
    print(f"  {'total':<43s}{total_params:>8d}{total_macs:>9d}"
          f"{sum(r['compares'] for r in rows):>10d}")
    print(f"\nParameters: {total_params}")
    print(f"MACs per window: {total_macs}")

    results = {
        "weights": args.weights.name,
        "test_windows": int(len(y)),
        "test_accuracy": accuracy,
        "per_class_accuracy": {n: float(a) for n, a in zip(har_data.CLASSES, per_class)},
        "confusion_matrix": {"classes": har_data.CLASSES,
                             "rows_true_cols_predicted": cm.tolist()},
        "parameters": total_params,
        "macs_per_window": total_macs,
        "layers": rows,
    }

    if not args.no_latency:
        cpu = cpu_name()
        lat = [measure_latency(model, x_raw, mean, std, threads=1)]
        print(f"\nBatch-1 CPU latency on {cpu} (standardise + forward, "
              f"{lat[0]['runs']} timed windows after {lat[0]['warmup']} warm-up):")
        for m in lat:
            print(f"  {m['threads']} thread: median {m['median_us']:.1f} us, "
                  f"mean {m['mean_us']:.1f} us, p99 {m['p99_us']:.1f} us")
        results["latency_batch1"] = {"cpu": cpu, "platform": platform.platform(),
                                     "torch": torch.__version__,
                                     "measurements": lat}

    summary_path = HERE / "train_summary.json"
    if summary_path.exists() and args.weights == HERE / "weights.npz":
        s = json.loads(summary_path.read_text())
        results["training"] = s
        print("\nSeeds (from train_summary.json):")
        for r in s["runs"]:
            mark = "  <- saved model" if r["seed"] == s["chosen_seed"] else ""
            print(f"  seed {r['seed']}: val {r['val_accuracy'] * 100:.2f}%, "
                  f"test {r['test_accuracy'] * 100:.2f}%{mark}")
        print(f"  test mean {s['test_accuracy_mean'] * 100:.2f}%, "
              f"std {s['test_accuracy_std'] * 100:.2f}, "
              f"range {s['test_accuracy_min'] * 100:.2f}-{s['test_accuracy_max'] * 100:.2f}%")

    if save_confusion_png(cm, HERE / "confusion_matrix.png"):
        results["confusion_matrix_png"] = "confusion_matrix.png"
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {HERE / 'results.json'}")


if __name__ == "__main__":
    main()
