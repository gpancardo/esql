#!/usr/bin/env python3
"""Clean and split the esql dataset.

Reads data/main.jsonl, cleans it (fixes malformed JSON, removes duplicates,
removes null/empty values), then splits into train/val/test (80-10-10)
stratified by task type.
"""

import json
import random
from collections import Counter

INPUT_FILE = "main.jsonl"
OUTPUT_TRAIN = "train.jsonl"
OUTPUT_VAL = "val.jsonl"
OUTPUT_TEST = "test.jsonl"
SEED = 42


def load_and_clean(filepath):
    """Load JSONL, attempt to fix malformed lines, return list of entries."""
    entries = []
    fixed = 0
    discarded = 0

    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(json.loads(stripped))
            except json.JSONDecodeError:
                # Attempt repair: common case is truncated "output": "
                try:
                    entry = json.loads(stripped + '"}')
                    entries.append(entry)
                    fixed += 1
                    print(f"  Fixed line {i}: appended closing quote + brace")
                except json.JSONDecodeError:
                    discarded += 1
                    print(f"  Discarded unfixable line {i}")

    if fixed:
        print(f"  Fixed {fixed} malformed line(s)")
    if discarded:
        print(f"  Discarded {discarded} unfixable line(s)")
    return entries


def remove_duplicates(entries):
    """Remove exact duplicates (all field values identical)."""
    seen = set()
    unique = []
    dup_count = 0
    for e in entries:
        key = json.dumps(e, sort_keys=True, ensure_ascii=False)
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)
            unique.append(e)
    return unique, dup_count


def remove_nulls(entries):
    """Remove entries where any required field is None or empty string."""
    required = ["task", "instruction", "input", "output"]
    clean = []
    removed = 0
    for e in entries:
        if e.get("task") is None or e.get("task") == "":
            removed += 1
        elif e.get("instruction") is None or e.get("instruction") == "":
            removed += 1
        elif e.get("input") is None or e.get("input") == "":
            removed += 1
        elif e.get("output") is None or e.get("output") == "":
            removed += 1
        else:
            clean.append(e)
    return clean, removed


def stratified_split(entries, stratify_key="task", train_ratio=0.8, val_ratio=0.1):
    """Split maintaining task distribution across splits."""
    random.seed(SEED)

    by_task = {}
    for e in entries:
        by_task.setdefault(e[stratify_key], []).append(e)

    train, val, test = [], [], []

    for task, items in by_task.items():
        random.shuffle(items)
        n = len(items)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        n_test = n - n_train - n_val

        # Guard against rounding producing negative test split
        if n_test < 0:
            n_val += n_test
            n_test = 0

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def write_jsonl(filepath, entries):
    """Write entries to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(entries)} entries to {filepath}")


def print_split_stats(train, val, test):
    """Print distribution stats per split."""
    for name, split in [("train", train), ("val", val), ("test", test)]:
        d = Counter(e["task"] for e in split)
        pcts = {k: f"{v / len(split) * 100:.1f}%" for k, v in d.items()}
        print(f"  {name}: {len(split)} entries — {pcts}")


def main():
    # 1. Load and fix malformed lines
    print("1. Loading main.jsonl ...")
    entries = load_and_clean(INPUT_FILE)
    print(f"   Loaded {len(entries)} valid entries\n")

    # 2. Remove duplicates
    print("2. Removing duplicates ...")
    entries, dup_count = remove_duplicates(entries)
    print(f"   Removed {dup_count} duplicates, {len(entries)} remain\n")

    # 3. Remove null/empty values
    print("3. Removing null/empty values ...")
    entries, null_count = remove_nulls(entries)
    print(f"   Removed {null_count} entries, {len(entries)} remain\n")

    # 4. Task distribution
    dist = Counter(e["task"] for e in entries)
    print(f"4. Task distribution: {dict(dist)}\n")

    # 5. Split 80-10-10
    print("5. Splitting (80-10-10, stratified by task) ...")
    train, val, test = stratified_split(entries)
    write_jsonl(OUTPUT_TRAIN, train)
    write_jsonl(OUTPUT_VAL, val)
    write_jsonl(OUTPUT_TEST, test)
    print()

    # 6. Verify
    print("6. Split summary:")
    print_split_stats(train, val, test)

    total_out = len(train) + len(val) + len(test)
    print(f"\n   Total: {total_out} (should match {len(entries)})")
    assert total_out == len(entries), "Split total != clean total!"

    print("\nDone.")


if __name__ == "__main__":
    main()
