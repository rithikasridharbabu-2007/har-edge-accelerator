"""Loading code for the UCI HAR raw inertial signals, shared by all Phase 1 scripts."""
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATASET_DIR = DATA_DIR / "UCI HAR Dataset"
CACHE_FILE = DATA_DIR / "har_raw.npz"

# Channel order used everywhere in this project (index 0..8).
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
# Class order (index 0..5). The dataset labels are 1..6; we subtract 1.
CLASSES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]
WINDOW = 128
EXPECTED = {"train": 7352, "test": 2947}


def _load_split_from_text(split):
    sig_dir = DATASET_DIR / split / "Inertial Signals"
    # Each file is (windows, 128). Stack to (windows, 9 channels, 128 samples).
    x = np.stack(
        [np.loadtxt(sig_dir / f"{c}_{split}.txt", dtype=np.float32) for c in CHANNELS],
        axis=1,
    )
    y = np.loadtxt(DATASET_DIR / split / f"y_{split}.txt", dtype=np.int64) - 1
    subject = np.loadtxt(DATASET_DIR / split / f"subject_{split}.txt", dtype=np.int64)
    return x, y, subject


def check_shapes(data):
    for split, n in EXPECTED.items():
        x, y, s = data[f"x_{split}"], data[f"y_{split}"], data[f"subject_{split}"]
        assert x.shape == (n, len(CHANNELS), WINDOW), (split, x.shape)
        assert y.shape == (n,) and s.shape == (n,), (split, y.shape, s.shape)
        assert set(np.unique(y)) == set(range(len(CLASSES))), np.unique(y)
        assert np.isfinite(x).all()
    # The official split separates volunteers.
    assert not set(data["subject_train"]) & set(data["subject_test"])


def build_cache():
    """Parse the text files once and store them as one .npz (much faster to reload)."""
    data = {}
    for split in ("train", "test"):
        x, y, s = _load_split_from_text(split)
        data[f"x_{split}"], data[f"y_{split}"], data[f"subject_{split}"] = x, y, s
    check_shapes(data)
    np.savez_compressed(CACHE_FILE, **data)
    return data


def load():
    """Return a dict with x_/y_/subject_ arrays for 'train' and 'test'."""
    if not CACHE_FILE.exists():
        if not DATASET_DIR.exists():
            raise SystemExit("Dataset not found. Run: python phase1_model/download_data.py")
        return build_cache()
    with np.load(CACHE_FILE) as f:
        data = {k: f[k] for k in f.files}
    check_shapes(data)
    return data
