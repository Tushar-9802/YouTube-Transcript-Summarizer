import json
from pathlib import Path

def merge_jsonl(files, out_path: Path, max_samples=None):
    merged = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                merged.append(json.loads(line))
    if max_samples:
        merged = merged[:max_samples]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Merged {len(merged)} samples into {out_path}")

if __name__ == "__main__":
    files = [
        "data/general/cnn.jsonl",
        "data/scientific/arxiv.jsonl",
        "data/scientific/pubmed.jsonl",
        "data/medical/meqsum.jsonl",
        "data/medical/ms2.jsonl",
    ]
    merge_jsonl([Path(f) for f in files], Path("data/multi_domain/train.jsonl"), max_samples=20000)
