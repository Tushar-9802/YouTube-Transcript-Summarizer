# scripts/main.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse, json
from yt_sum.utils.config import Config
from yt_sum.utils.logging import get_logger
from yt_sum.pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser(description="YouTube → Transcript → Summary")
    p.add_argument("--url", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--whisper_size", default=None, help="Override Faster-Whisper size; None = auto")
    p.add_argument("--summarizer_model", default="facebook/bart-large-cnn")
    p.add_argument("--no_chunk", action="store_true")
    p.add_argument("--language", default=None)
    p.add_argument("--prefer_accuracy", action="store_true")
    args = p.parse_args()

    cfg = Config(args.config); cfg.ensure_dirs()
    log = get_logger(level=cfg.log_level)

    def cb(stage, pct, extra):
        names = {"init":"Init","download":"Step 1/4 Download","transcribe":"Step 2/4 Transcribe","summarize":"Step 3/4 Summarize"}
        if stage == "init":
            d = extra.get("device", {}); log.info(f"Device: CUDA={d.get('cuda')} | {d.get('name')} | Free {d.get('free_gb')} / {d.get('total_gb')} GB")
        elif stage == "transcribe" and pct == 10:
            a = extra.get("asr_choice", {}); log.info(f"ASR auto-choice: backend={a.get('backend')} size={a.get('model_size')} compute={a.get('compute_type')} lang={a.get('language')}")
        elif stage == "summarize" and pct == 10:
            s = extra.get("summarizer", {}); log.info(f"Summarizer: {s.get('model')} | chunked={s.get('chunked')} | chunk={s.get('chunk_tokens')}/{s.get('chunk_overlap')} | beams={s.get('num_beams')} | perchunk={s.get('per_chunk_min')}-{s.get('per_chunk_max')} | fuse={s.get('fuse_min')}-{s.get('fuse_max')}")
        log.info(f"{names.get(stage, stage)}… {pct}%")

    meta, res = run_pipeline(
        url=args.url,
        cfg=cfg,
        whisper_size=args.whisper_size,
        summarizer_model=args.summarizer_model,
        chunked=not args.no_chunk,
        language=args.language,
        prefer_accuracy=args.prefer_accuracy,
        progress=cb,
    )

    out = {"meta": meta, "results": res}
    out_path = cfg.outputs_dir / f"{meta['video']['id']}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"✅ Done. Summary saved to: {out_path}\n\n--- SUMMARY ---\n{res['summary']}")

if __name__ == "__main__":
    main()
