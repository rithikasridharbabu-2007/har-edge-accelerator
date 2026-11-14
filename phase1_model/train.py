"""Train the Phase 1 CNN on the raw UCI HAR signals.

    python phase1_model/train.py            # seeds 0 1 2
    python phase1_model/train.py --seeds 0  # one seed

Model selection uses ONLY a validation set held out from the official train
split by subject. The official test set is evaluated once per seed, after the
checkpoint for that seed has been fixed, purely for reporting.

Outputs:
  phase1_model/runs/seed<N>.npz      per-seed best-validation weights (git-ignored)
  phase1_model/weights.npz           the seed with the best validation accuracy
  phase1_model/train_summary.json    per-seed validation/test numbers
and then runs evaluate.py on weights.npz (results.json, confusion_matrix.png).
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import har_data
from model import HarCnn, count_parameters, save_npz, standardise

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"

# Volunteers held out of the train split for validation (fixed, chosen before
# any training; 4 of the 21 training volunteers).
VAL_SUBJECTS = (7, 15, 22, 28)

EPOCHS = 60
BATCH_SIZE = 64
LR = 2e-3
WEIGHT_DECAY = 1e-4


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y).item()
    acc = (logits.argmax(1) == y).float().mean().item()
    return loss, acc


def train_one_seed(seed, data):
    set_seed(seed)
    val_mask = np.isin(data["subject_train"], VAL_SUBJECTS)
    x_fit, y_fit = data["x_train"][~val_mask], data["y_train"][~val_mask]
    x_val, y_val = data["x_train"][val_mask], data["y_train"][val_mask]

    # Per-channel statistics from the fitting part of the train split only.
    mean = x_fit.mean(axis=(0, 2))
    std = x_fit.std(axis=(0, 2))

    def prep(x):
        return torch.from_numpy(standardise(x, mean, std).astype(np.float32))

    x_fit_t, y_fit_t = prep(x_fit), torch.from_numpy(y_fit)
    x_val_t, y_val_t = prep(x_val), torch.from_numpy(y_val)
    x_test_t, y_test_t = prep(data["x_test"]), torch.from_numpy(data["y_test"])

    model = HarCnn()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    gen = torch.Generator().manual_seed(seed)

    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": -1, "state": None}
    n = len(x_fit_t)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            loss = nn.functional.cross_entropy(model(x_fit_t[idx]), y_fit_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        val_loss, val_acc = evaluate(model, x_val_t, y_val_t)
        # Keep the epoch with the best validation accuracy (ties: lower val loss).
        if (val_acc, -val_loss) > (best["val_acc"], -best["val_loss"]):
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  seed {seed} epoch {epoch:3d}  val loss {val_loss:.4f}  "
                  f"val acc {val_acc * 100:.2f}%")

    model.load_state_dict(best["state"])
    _, test_acc = evaluate(model, x_test_t, y_test_t)   # reporting only
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"seed{seed}.npz"
    save_npz(path, model, mean, std)
    print(f"  seed {seed}: best epoch {best['epoch']}, val acc "
          f"{best['val_acc'] * 100:.2f}%, test acc {test_acc * 100:.2f}%")
    return {
        "seed": seed, "best_epoch": best["epoch"],
        "val_accuracy": best["val_acc"], "val_loss": best["val_loss"],
        "test_accuracy": test_acc,
        "n_fit": int(len(x_fit)), "n_val": int(len(x_val)),
        "parameters": count_parameters(model),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no-evaluate", action="store_true",
                    help="skip the evaluate.py report at the end")
    args = ap.parse_args()

    data = har_data.load()
    t0 = time.time()
    runs = [train_one_seed(s, data) for s in args.seeds]

    # Choose the saved model by validation accuracy (ties: lower validation loss).
    chosen = max(runs, key=lambda r: (r["val_accuracy"], -r["val_loss"]))
    shutil.copyfile(RUNS_DIR / f"seed{chosen['seed']}.npz", HERE / "weights.npz")

    test = np.array([r["test_accuracy"] for r in runs])
    summary = {
        "val_subjects": list(VAL_SUBJECTS),
        "hyperparameters": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
                            "weight_decay": WEIGHT_DECAY, "optimizer": "Adam",
                            "schedule": "cosine"},
        "runs": runs,
        "chosen_seed": chosen["seed"],
        "test_accuracy_mean": float(test.mean()),
        "test_accuracy_std": float(test.std()),   # population std over the seeds
        "test_accuracy_min": float(test.min()),
        "test_accuracy_max": float(test.max()),
    }
    (HERE / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nTrained {len(runs)} seed(s) in {time.time() - t0:.0f} s")
    print(f"Test accuracy over seeds: mean {test.mean() * 100:.2f}%, "
          f"std {test.std() * 100:.2f}, min {test.min() * 100:.2f}%, "
          f"max {test.max() * 100:.2f}%")
    print(f"Chosen by validation accuracy: seed {chosen['seed']} "
          f"(test acc {chosen['test_accuracy'] * 100:.2f}%) -> weights.npz")
    print(f"Parameters: {chosen['parameters']}")

    # Full report for the saved model (confusion matrix, MACs, latency, results.json).
    if not args.no_evaluate:
        print("\n--- evaluate.py on weights.npz ---")
        import evaluate
        evaluate.main([])


if __name__ == "__main__":
    main()
