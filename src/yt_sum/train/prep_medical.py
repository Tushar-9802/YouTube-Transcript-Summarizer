# src/yt_sum/train/prep_medical.py
import argparse
import json
import random
from pathlib import Path
from tqdm import tqdm


def iter_jsonl_or_json(path: Path):
    """
    Stream records from a JSONL or JSON file.
    - JSONL: either line-delimited JSON, or sometimes a JSON array mislabeled as .jsonl
    - JSON: list of objects or dict with nested arrays.
    """
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            try:
                # Try loading entire file (covers JSON array mislabeled as .jsonl)
                obj = json.load(f)
                if isinstance(obj, list):
                    for x in obj:
                        yield x
                    return
            except Exception:
                # Fallback: line-by-line JSON
                f.seek(0)
                for line in f:
                    if line.strip():
                        yield json.loads(line)

        elif path.suffix == ".json":
            try:
                obj = json.load(f)
                if isinstance(obj, list):
                    for x in obj:
                        yield x
                elif isinstance(obj, dict):
                    for k, v in obj.items():
                        if isinstance(v, list):
                            for x in v:
                                yield x
            except Exception:
                return

def extract_text_summary(obj):
    if not isinstance(obj, dict):
        return None, None

    text, summary = None, None

    # Use abstract first
    if "abstract" in obj:
        if isinstance(obj["abstract"], str):
            text = obj["abstract"]
        elif isinstance(obj["abstract"], list):
            text = " ".join([x for x in obj["abstract"] if isinstance(x, str)])
        elif isinstance(obj["abstract"], dict):
            text = " ".join([str(v) for v in obj["abstract"].values() if isinstance(v, str)])

    # Fallback: structured_abstract
    if not text and "structured_abstract" in obj:
        if isinstance(obj["structured_abstract"], list):
            text = " ".join([x.get("text", "") for x in obj["structured_abstract"] if isinstance(x, dict)])

    # Summary extraction
    if "summary" in obj and isinstance(obj["summary"], str):
        summary = obj["summary"]
    elif "review_summary" in obj:
        summary = obj["review_summary"]
    elif "target" in obj:
        summary = obj["target"]
    elif "title" in obj:  # last fallback
        summary = obj["title"]

    return text, summary




def prepare_medical(ms2_dir: Path, out_dir: Path, max_samples: int = 5000):
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "medical.jsonl"

    reservoir = []
    n = 0

    files = list(ms2_dir.rglob("*.json")) + list(ms2_dir.rglob("*.jsonl"))
    print(f"📂 Found {len(files)} raw MS² files")

    for file in tqdm(files, desc="Streaming MS²"):
        for i, obj in enumerate(iter_jsonl_or_json(file)):
            if i < 3:  # show first few keys per file
                print("DEBUG sample keys:", list(obj.keys()))
            text, summary = extract_text_summary(obj)
            if text and summary:
                n += 1
                if len(reservoir) < max_samples:
                    reservoir.append({"text": text, "summary": summary})
                else:
                    j = random.randint(0, n - 1)
                    if j < max_samples:
                        reservoir[j] = {"text": text, "summary": summary}

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in reservoir:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Saved {len(reservoir)} samples to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms2_dir", type=str, required=True, help="Path to MS² extracted folder")
    parser.add_argument("--out_dir", type=str, required=True, help="Where to save processed dataset")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    prepare_medical(Path(args.ms2_dir), Path(args.out_dir), args.max_samples)


if __name__ == "__main__":
    main()
