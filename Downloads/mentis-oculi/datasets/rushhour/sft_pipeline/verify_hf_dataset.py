"""
Sanity-check an HF dataset saved by build_hf_dataset.py.

Usage:
    python verify_hf_dataset.py --path ../hf_rushhour
"""
import argparse
from pathlib import Path

from datasets import load_from_disk


def short(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    ds = load_from_disk(str(Path(args.path).resolve()))
    print(f"Splits: {list(ds.keys())}")
    for split, d in ds.items():
        print(f"  {split}: {len(d)} rows")

    first = ds[list(ds.keys())[0]]
    print("\nFeatures:")
    for k, v in first.features.items():
        print(f"  {k}: {v}")

    print("\nFirst 3 rows:")
    for i in range(min(3, len(first))):
        row = first[i]
        qi = row["question_interleave"]
        si = row["solution_interleave"]
        qimgs = row["question_images"]
        simgs = row["solution_images"]

        n_qi_img = sum(1 for it in qi if it["type"] == "image")
        n_si_img = sum(1 for it in si if it["type"] == "image")
        assert n_qi_img == len(qimgs), f"row {i}: q interleave imgs {n_qi_img} != question_images {len(qimgs)}"
        assert n_si_img == len(simgs), f"row {i}: s interleave imgs {n_si_img} != solution_images {len(simgs)}"

        print(f"\n--- row {i} ---")
        print(f"  id: {row['id']}")
        print(f"  knowledge/subknowledge: {row['knowledge']} / {row['subknowledge']}")
        print(f"  answer: {short(row['answer'])}")
        print(f"  question_interleave ({len(qi)}):")
        for it in qi:
            print(f"    [{it['type']} #{it['index']}] {short(it['content'], 100)}")
        print(f"  question_images ({len(qimgs)}): {[f'{im.size[0]}x{im.size[1]}' for im in qimgs]}")
        print(f"  solution_interleave ({len(si)}):")
        for it in si:
            print(f"    [{it['type']} #{it['index']}] {short(it['content'], 100)}")
        print(f"  solution_images ({len(simgs)}): {[f'{im.size[0]}x{im.size[1]}' for im in simgs]}")

    print("\nAll invariants hold.")


if __name__ == "__main__":
    main()
