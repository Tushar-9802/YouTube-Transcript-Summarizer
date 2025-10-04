# src/yt_sum/train/prep_engineering.py
from __future__ import annotations
import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, Dict

from datasets import load_dataset
from tqdm import tqdm


# Broad engineering keyword net (title + first ~2k chars of article)
ENG_KEYWORDS = [
    # general
    r"\bengineering\b", r"\bengineer(s|ing)?\b", r"\bapplied\s+science\b",
    # mechanical / aero
    r"\bmechanical\b", r"\baero(space|nautical)?\b", r"\bthermo(dynamics)?\b",
    r"\bfluid(s)?\b", r"\bCFD\b", r"\bturbulen(ce|t)\b", r"\bcombust(ion|or)\b",
    r"\bfinite\s+element\b", r"\bFEM\b", r"\bstructural\b", r"\bmaterials?\b",
    r"\bmanufactur(ing|e)\b", r"\badditive\s+manufacturing\b", r"\b3D\s*print(ing)?\b",
    r"\brobot(ics|ic|)\b", r"\bkinematic(s)?\b", r"\bmechatronic(s)?\b",
    # electrical / electronics / signal / embedded
    r"\belectrical\b", r"\belectronic(s)?\b", r"\bVLSI\b", r"\bsemiconductor(s)?\b",
    r"\bintegrated\s+circuits?\b", r"\bICs?\b", r"\bFPGA\b", r"\bASIC\b",
    r"\bantenna(s)?\b", r"\bmicrowave(s)?\b", r"\bmmWave\b",
    r"\bpower\s+(system|electronics)\b", r"\binverter(s)?\b", r"\bconverter(s)?\b",
    r"\bcontrol(s)?\b", r"\bPID\b", r"\bstate(?:\s*space)?\b", r"\bLQR\b",
    r"\bembedded\b", r"\breal-?time\b", r"\bfirmware\b",
    r"\bsignal\s+processing\b", r"\bDSP\b", r"\bfilter(s|ing)?\b",
    # civil / structural / geotech / transport
    r"\bcivil\b", r"\bgeotechnical\b", r"\bgeotech\b", r"\bseismic\b",
    r"\bearthquake(s)?\b", r"\bbridge(s)?\b", r"\bstructur(al|e)\b",
    r"\bconcrete\b", r"\basphalt\b", r"\bpavement(s)?\b",
    r"\btransport(ation)?\b", r"\btraffic\b",
    # chemical / process / energy
    r"\bchemical\b", r"\bprocess\s+engineering\b", r"\breactor\b",
    r"\bcatalyst(s|ic)?\b", r"\bmembrane(s)?\b", r"\bheat\s+exchanger\b",
    r"\bbattery\b", r"\bfuel\s+cell(s)?\b", r"\bphotovoltaic(s)?\b", r"\bsolar\b",
    # industrial / reliability
    r"\bindustrial\b", r"\boperations\s+research\b", r"\boptimization\b",
    r"\breliability\b", r"\bmaintenance\b",
]

ENG_RE = re.compile("|".join(ENG_KEYWORDS), flags=re.IGNORECASE)


def is_engineering(title: str, article: str) -> bool:
    if not title and not article:
        return False
    # Search title first (strong signal), then body prefix to keep compute cheap
    if title and ENG_RE.search(title):
        return True
    prefix = (article or "")[:2000]
    return bool(ENG_RE.search(prefix))


def sample_and_convert(ds, max_samples: int, min_article_chars: int = 800) -> Iterable[Dict]:
    """
    From ccdv/arxiv-summarization (fields: 'article', 'abstract', 'article_id'),
    filter to engineering-like docs and yield JSONL rows {'text','summary'}.
    """
    # Shuffle for diversity
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    kept = 0
    for i in idxs:
        ex = ds[i]
        article = (ex.get("article") or "").strip()
        abstract = (ex.get("abstract") or "").strip()
        title = ""  # dataset doesn't always have title; often embedded at top of article

        # Basic quality gates
        if len(article) < min_article_chars:
            continue
        if not abstract:
            continue

        if is_engineering(title, article):
            yield {
                "text": article,
                "summary": abstract,
                "source": "ccdv/arxiv-summarization",
                "id": ex.get("article_id", f"idx_{i}"),
            }
            kept += 1
            if kept >= max_samples:
                break


def main():
    parser = argparse.ArgumentParser(description="Prepare Engineering JSONL from arXiv (engineering-like subset).")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "engineering.jsonl"

    print("⚡ Preparing Engineering Dataset (from ccdv/arxiv-summarization)")

    # Load maintained HF dataset (no deprecated scripts)
    # Splits: train/validation/test exist; train is largest.
    ds = load_dataset("ccdv/arxiv-summarization", split="train")

    # Oversample pool so filter has room; cap to dataset size
    pool_n = min(len(ds), max(args.max_samples * 10, args.max_samples + 1000))

    print(f"  Loaded {len(ds)} training docs. Filtering for engineering keywords...")
    kept = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(sample_and_convert(ds.select(range(pool_n)), args.max_samples), total=args.max_samples):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

    print(f"✅ Saved {kept} samples to {out_path}")


if __name__ == "__main__":
    main()
