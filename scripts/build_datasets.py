"""Rebuild the classification datasets from their public sources into data/datasets/.

Each dataset becomes a binary question about one native class, drawn with a fixed seed and truncated at a fixed number
of characters (flipjev/datasets.py). The Hugging Face datasets are pinned to the revisions that were used, so the
rebuilt cases.jsonl is byte-identical to the original; the script prints each file's SHA-256 so it can be checked
against data/datasets/checksums.json. The arXiv abstracts come from the live arXiv API and ship with the repository.

    python -m scripts.build_datasets                      # all nine datasets (and IMDB with --imdb)
    python -m scripts.build_datasets --datasets trec ag_news
    python -m scripts.build_datasets --check              # rebuild and compare against checksums.json
"""

import argparse
import json
from pathlib import Path

from flipjev import datasets as D


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=D.TOPIC_DATASETS,
                        help="which datasets to build (default: the nine classification datasets)")
    parser.add_argument("--imdb", action="store_true", help="also build the IMDB sentiment dataset (reported separately)")
    parser.add_argument("--out", type=Path, default=D.DATA)
    parser.add_argument("--check", action="store_true", help="compare each rebuilt file against data/datasets/checksums.json")
    args = parser.parse_args()
    names = list(args.datasets) + (["imdb"] if args.imdb else [])
    stored = {}
    checksums = D.DATA / "checksums.json"
    if args.check and checksums.exists():
        stored = json.loads(checksums.read_text())
    ok = True
    for name in names:
        if name == "arxiv_abstracts" and D.REGISTRY[name]["revision"] is None:
            print(f"{name:16s} comes from the live arXiv API; the stored copy ships with the repository, skipping")
            continue
        meta = D.build(name, out_dir=args.out)
        line = f"{name:16s} labels={meta['labels']} sha256={meta['dataset_sha256'][:16]}"
        if args.check and name in stored:
            match = meta["dataset_sha256"] == stored[name]
            ok = ok and match
            line += "  " + ("matches" if match else f"DIFFERS from {stored[name][:16]}")
        print(line)
    if args.check:
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
