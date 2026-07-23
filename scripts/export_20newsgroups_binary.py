#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.datasets import dump_svmlight_file, fetch_20newsgroups_vectorized


DEFAULT_CLASS_A = "comp.graphics"
DEFAULT_CLASS_B = "sci.space"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a binary 20NewsGroup subset as train/test SVMlight files."
    )
    parser.add_argument("--class-a", default=DEFAULT_CLASS_A)
    parser.add_argument("--class-b", default=DEFAULT_CLASS_B)
    parser.add_argument("--out-dir", type=Path, default=Path("data/20newsgroup"))
    parser.add_argument("--prefix", default="20newsgroup")
    parser.add_argument("--list-categories", action="store_true")
    return parser


def get_category_names() -> List[str]:
    dataset = fetch_20newsgroups_vectorized(subset="train")
    return list(dataset.target_names)


def resolve_category(name: str, category_names: List[str]) -> int:
    if name not in category_names:
        raise SystemExit(
            "Unknown category '{}'. Available categories: {}".format(
                name, ", ".join(category_names)
            )
        )
    return category_names.index(name)


def extract_binary_subset(dataset, class_a_index: int, class_b_index: int) -> Tuple[object, np.ndarray]:
    mask = np.logical_or(dataset.target == class_a_index, dataset.target == class_b_index)
    subset_x = dataset.data[mask]
    subset_y = (dataset.target[mask] == class_b_index).astype(np.int32)
    return subset_x, subset_y


def write_metadata(
    meta_path: Path,
    class_a: str,
    class_b: str,
    train_rows: int,
    test_rows: int,
    feature_count: int,
) -> None:
    lines = [
        "source_dataset=20NewsGroup",
        "label_0={}".format(class_a),
        "label_1={}".format(class_b),
        "train_rows={}".format(train_rows),
        "test_rows={}".format(test_rows),
        "feature_count={}".format(feature_count),
    ]
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    category_names = get_category_names()
    if args.list_categories:
        for name in category_names:
            print(name)
        return

    if args.class_a == args.class_b:
        raise SystemExit("--class-a and --class-b must be different.")

    class_a_index = resolve_category(args.class_a, category_names)
    class_b_index = resolve_category(args.class_b, category_names)

    train_dataset = fetch_20newsgroups_vectorized(subset="train")
    test_dataset = fetch_20newsgroups_vectorized(subset="test")

    train_x, train_y = extract_binary_subset(train_dataset, class_a_index, class_b_index)
    test_x, test_y = extract_binary_subset(test_dataset, class_a_index, class_b_index)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "{}_train.svm".format(args.prefix)
    test_path = args.out_dir / "{}_test.svm".format(args.prefix)
    meta_path = args.out_dir / "{}_meta.txt".format(args.prefix)

    with train_path.open("wb") as train_file:
        dump_svmlight_file(train_x, train_y, train_file, zero_based=False)
    with test_path.open("wb") as test_file:
        dump_svmlight_file(test_x, test_y, test_file, zero_based=False)

    write_metadata(
        meta_path,
        args.class_a,
        args.class_b,
        int(train_x.shape[0]),
        int(test_x.shape[0]),
        int(train_x.shape[1]),
    )

    print("wrote {}".format(train_path))
    print("wrote {}".format(test_path))
    print("wrote {}".format(meta_path))
    print("label 0 -> {}".format(args.class_a))
    print("label 1 -> {}".format(args.class_b))
    print("train rows: {}".format(train_x.shape[0]))
    print("test rows: {}".format(test_x.shape[0]))
    print("feature count: {}".format(train_x.shape[1]))


if __name__ == "__main__":
    main()
