"""Download and unpack the UCI HAR dataset into the git-ignored data/ folder.

    python phase1_model/download_data.py

The UCI archive is a zip that contains another zip ("UCI HAR Dataset.zip").
"""
import io
import urllib.request
import zipfile

import har_data

URL = ("https://archive.ics.uci.edu/static/public/240/"
       "human+activity+recognition+using+smartphones.zip")
OUTER_ZIP = har_data.DATA_DIR / "uci_har.zip"
INNER_NAME = "UCI HAR Dataset.zip"


def main():
    har_data.DATA_DIR.mkdir(exist_ok=True)

    if not har_data.DATASET_DIR.exists():
        if not OUTER_ZIP.exists():
            print(f"Downloading {URL}")
            tmp = OUTER_ZIP.with_suffix(".part")
            urllib.request.urlretrieve(URL, tmp)
            tmp.rename(OUTER_ZIP)
        print(f"Unpacking {OUTER_ZIP.name} ({OUTER_ZIP.stat().st_size / 1e6:.1f} MB)")
        with zipfile.ZipFile(OUTER_ZIP) as outer:
            inner_bytes = outer.read(INNER_NAME)
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
            members = [m for m in inner.namelist() if m.startswith("UCI HAR Dataset/")]
            inner.extractall(har_data.DATA_DIR, members)
    else:
        print(f"Already unpacked: {har_data.DATASET_DIR}")

    data = har_data.build_cache()  # also verifies shapes
    for split in ("train", "test"):
        x, s = data[f"x_{split}"], data[f"subject_{split}"]
        print(f"{split}: x {x.shape} {x.dtype}, y {data[f'y_{split}'].shape}, "
              f"{len(set(s))} subjects")
    print(f"Shapes OK. Cached to {har_data.CACHE_FILE}")


if __name__ == "__main__":
    main()
