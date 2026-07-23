#!/usr/bin/env python3
"""Export full 20NewsGroup as binary classification (first 10 vs last 10 categories)."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.datasets import dump_svmlight_file, fetch_20newsgroups_vectorized


def main():
    parser = argparse.ArgumentParser(
        description="Export full 20NewsGroup (all 20 categories) as a binary split."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/20newsgroup_full"))
    parser.add_argument("--prefix", default="20newsgroup_full")
    args = parser.parse_args()

    train_ds = fetch_20newsgroups_vectorized(subset="train")
    test_ds = fetch_20newsgroups_vectorized(subset="test")

    # First 10 categories (alphabetical indices 0-9) -> label 0
    # Last 10 categories (alphabetical indices 10-19) -> label 1
    train_y = (train_ds.target >= 10).astype(np.int32)
    test_y = (test_ds.target >= 10).astype(np.int32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "{}_train.svm".format(args.prefix)
    test_path = args.out_dir / "{}_test.svm".format(args.prefix)
    meta_path = args.out_dir / "{}_meta.txt".format(args.prefix)

    with train_path.open("wb") as f:
        dump_svmlight_file(train_ds.data, train_y, f, zero_based=False)
    with test_path.open("wb") as f:
        dump_svmlight_file(test_ds.data, test_y, f, zero_based=False)

    group_a = [train_ds.target_names[i] for i in range(10)]
    group_b = [train_ds.target_names[i] for i in range(10, 20)]

    meta_lines = [
        "source_dataset=20NewsGroup_full_binary",
        "label_0={}".format(",".join(group_a)),
        "label_1={}".format(",".join(group_b)),
        "train_rows={}".format(train_ds.data.shape[0]),
        "test_rows={}".format(test_ds.data.shape[0]),
        "feature_count={}".format(train_ds.data.shape[1]),
    ]
    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    print("wrote {}".format(train_path))
    print("wrote {}".format(test_path))
    print("wrote {}".format(meta_path))
    print("label 0 ({} cats): {}".format(len(group_a), ", ".join(group_a)))
    print("label 1 ({} cats): {}".format(len(group_b), ", ".join(group_b)))
    print("train rows: {}".format(train_ds.data.shape[0]))
    print("test rows: {}".format(test_ds.data.shape[0]))
    print("feature count: {}".format(train_ds.data.shape[1]))


if __name__ == "__main__":
    main()
