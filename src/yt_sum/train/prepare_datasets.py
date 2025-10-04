from datasets import load_dataset
import json
import argparse
from pathlib import Path

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def prepare(dataset_name: str, out_dir: Path, max_samples: int = None):
    if dataset_name == "cnn":
        ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
        mapped = [{"text": x["article"], "summary": x["highlights"]} for x in ds]
    elif dataset_name == "arxiv":
        ds = load_dataset("ccdv/arxiv-summarization", split="train")
        mapped = [{"text": x["article"], "summary": x["abstract"]} for x in ds]
    elif dataset_name == "pubmed":
        ds = load_dataset("ccdv/pubmed-summarization", split="train")
        mapped = [{"text": x["article"], "summary": x["abstract"]} for x in ds]
    elif dataset_name == "meqsum":
        ds = load_dataset("abisee/meqsum", split="train")
        mapped = [{"text": x["question"], "summary": x["answer"]} for x in ds]
    elif dataset_name == "ms2":
        ds = load_dataset("ms2/ms2", split="train")
        mapped = [{"text": x["source"], "summary": x["target"]} for x in ds]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if max_samples:
        mapped = mapped[:max_samples]

    out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(mapped, out_dir / f"{dataset_name}.jsonl")
    print(f"✅ Saved {len(mapped)} samples to {out_dir}/{dataset_name}.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="cnn/arxiv/pubmed/meqsum/ms2")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    prepare(args.dataset, Path(args.out_dir), args.max_samples)
