#!/usr/bin/env python3
"""Compute dataset metrics for HuggingFace publication.

Reads train.jsonl, val.jsonl, test.jsonl and outputs metrics.json.
"""

import json
from collections import Counter

SPLITS = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


def load_jsonl(filepath):
    entries = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_stats(values):
    """Compute summary statistics for a list of numeric values."""
    values = sorted(values)
    n = len(values)
    if n == 0:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "p5": 0, "p95": 0}
    return {
        "count": n,
        "min": values[0],
        "max": values[-1],
        "mean": round(sum(values) / n, 1),
        "median": values[n // 2],
        "p5": values[max(0, n * 5 // 100)],
        "p95": values[min(n - 1, n * 95 // 100)],
    }


def main():
    all_entries = []
    metrics = {}

    for split_name, filepath in SPLITS.items():
        entries = load_jsonl(filepath)
        all_entries.extend(entries)

        task_dist = Counter(e["task"] for e in entries)

        split_metrics = {
            "num_entries": len(entries),
            "task_distribution": dict(task_dist),
        }

        for field in ["instruction", "input", "output"]:
            char_lengths = [len(e[field]) for e in entries]
            word_lengths = [len(e[field].split()) for e in entries]
            split_metrics[f"{field}_char_length"] = compute_stats(char_lengths)
            split_metrics[f"{field}_word_length"] = compute_stats(word_lengths)

        metrics[split_name] = split_metrics

    # Overall
    total = len(all_entries)
    overall_task_dist = Counter(e["task"] for e in all_entries)
    metrics["overall"] = {
        "total_entries": total,
        "task_distribution": dict(overall_task_dist),
        "task_distribution_pct": {
            k: round(v / total * 100, 1) for k, v in overall_task_dist.items()
        },
    }

    # Vocabulary
    all_text = " ".join(
        " ".join([e["instruction"], e["input"], e["output"]]) for e in all_entries
    )
    words = all_text.split()
    lower_words = set(w.lower() for w in words)
    metrics["vocabulary"] = {
        "total_words": len(words),
        "unique_words": len(set(words)),
        "unique_words_lowercase": len(lower_words),
    }

    # Unique values per field (useful for understanding diversity)
    for field in ["task", "instruction", "input", "output"]:
        values = [e[field] for e in all_entries]
        metrics["overall"][f"unique_{field}s"] = len(set(values))
        metrics["overall"][f"total_{field}s"] = len(values)

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
