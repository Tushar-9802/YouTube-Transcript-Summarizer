# factcheck.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_DEF_MODELS = [
    "MoritzLaurer/deberta-v3-base-zeroshot-v2",
    "facebook/bart-large-mnli"
]

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

class EntailmentScorer:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        names = [self.model_name] if self.model_name else _DEF_MODELS
        last_err = None
        for name in names:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name)
                self.model = AutoModelForSequenceClassification.from_pretrained(name)
                if torch.cuda.is_available():
                    try:
                        self.model = self.model.half().to("cuda")
                    except Exception:
                        self.model = self.model.to("cuda")
                self.model_name = name
                return
            except Exception as e:
                last_err = e
        raise RuntimeError(f"Failed to load NLI model: {last_err}")

    def _probs(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits.float()
        if logits.shape[-1] == 3:
            import torch.nn.functional as F
            probs = F.softmax(logits[0], dim=-1).tolist()
            c, n, e = probs[0], probs[1], probs[2]
            return e, n, c
        else:
            import torch.nn.functional as F
            p = torch.sigmoid(logits[0]).mean().item()
            return p, 1.0 - p, 0.0

    def score_summary_against_transcript(self, summary: str, transcript: str, *, k_support: int = 3) -> Dict[str, Any]:
        sum_sents = _SENT_SPLIT.split((summary or "").strip())
        tr_sents = _SENT_SPLIT.split((transcript or "").strip())
        tr_sets = [set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", s.lower())) for s in tr_sents]

        def top_supporters(hyp: str) -> List[int]:
            hset = set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", hyp.lower()))
            scores = []
            for idx, tset in enumerate(tr_sets):
                if not tset:
                    continue
                inter = len(hset & tset)
                denom = (len(hset) ** 0.5) * (len(tset) ** 0.5)
                sim = inter / denom if denom else 0.0
                if sim > 0:
                    scores.append((sim, idx))
            scores.sort(key=lambda x: x[0], reverse=True)
            return [i for (_s, i) in scores[:k_support]]

        sent_scores: List[float] = []
        details: List[Dict[str, Any]] = []
        for s in sum_sents:
            s = s.strip()
            if not s:
                continue
            idxs = top_supporters(s)
            if not idxs:
                sent_scores.append(0.0)
                details.append({"sentence": s, "confidence": 0.0, "supports": []})
                continue
            e_scores = []
            supports = []
            for i in idxs:
                e, n, c = self._probs(tr_sents[i], s)
                e_scores.append(e)
                supports.append({"evidence": tr_sents[i], "entailment": float(e)})
            conf = float(sum(e_scores) / len(e_scores))
            sent_scores.append(conf)
            details.append({"sentence": s, "confidence": conf, "supports": supports})
        overall = float(sum(sent_scores) / max(1, len(sent_scores)))
        return {"overall_confidence": overall, "sentences": details}
