"""Recompute the chapter's Table 1, Table 3, and failure counts from saved outputs. Usage: python recompute_chapter_tables.py <repo_root> [out.json]"""
import json, math, re, sys
from pathlib import Path
import numpy as np, pandas as pd, spacy, torch
from bert_score import score
from scipy import stats
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = Path(sys.argv[1])
DOMAINS = ["medical", "engineering", "scientific"]
FAILED_ASR = {"j3mhkYbznBk", "LBUFrBHAjcs", "EjaGktVQdNg", "iUjVnV5FsYo"}
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(ROOT / "evaluation/results/evaluation/detailed_results.csv")
CRLF, LF = chr(13) + chr(10), chr(10)  # git autocrlf on Windows otherwise shifts GPT-2 and BERTScore values
df["summary"] = df["summary"].fillna("").str.replace(CRLF, LF, regex=False)
T = {}
for d in DOMAINS:
    for line in open(ROOT / f"evaluation/data/youtube_test/{d}_test.jsonl", encoding="utf-8"):
        r = json.loads(line); T[r["video_id"]] = r
b = df[df.model_type == "base"].set_index("video_id")
l = df[df.model_type == "lora"].set_index("video_id")
ids = [i for i in b.index if i in l.index]
refs = [T[i]["reference_summary"] for i in ids]
_, _, Fb = score(b.loc[ids, "summary"].tolist(), refs, lang="en", device=dev, verbose=False)
_, _, Fl = score(l.loc[ids, "summary"].tolist(), refs, lang="en", device=dev, verbose=False)
b.loc[ids, "bert"] = Fb.numpy().astype(float); l.loc[ids, "bert"] = Fl.numpy().astype(float)

def pair(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    pooled = math.sqrt((x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2)
    return dict(base=x.mean(), lora=y.mean(), sd_base=x.std(ddof=1), sd_lora=y.std(ddof=1), delta=100 * (y.mean() - x.mean()) / x.mean(),
                p=float(stats.ttest_rel(y, x)[1]), d=abs(x.mean() - y.mean()) / pooled)

out = {"table1": {}, "table3": {}, "counts": {}, "clean": {}}
for d in DOMAINS:
    k = [i for i in ids if T[i]["domain"] == d]
    for m, n in [("rouge1_f", "R-1"), ("rouge2_f", "R-2"), ("rougeL_f", "R-L"), ("bert", "BERT")]:
        out["table1"][f"{d}/{n}"] = pair(b.loc[k, m], l.loc[k, m])

nlp = spacy.load("en_core_web_sm")
NOM = re.compile(r"(tion|sion|ment|ness|ance|ence|ity|ism)s?$", re.I)
def style(t):
    doc = nlp(t); sents = [s for s in doc.sents if len(s) > 1]
    pas = sum(any(k.dep_ in ("nsubjpass", "auxpass", "csubjpass") for k in s) for s in sents)
    nouns = [k for k in doc if k.pos_ == "NOUN"]
    return 100 * pas / max(len(sents), 1), 100 * sum(bool(NOM.search(k.text)) and len(k.text) > 5 for k in nouns) / max(len(nouns), 1)
sb = np.array([style(t) for t in b.loc[ids, "summary"]]); sl = np.array([style(t) for t in l.loc[ids, "summary"]])
out["table3"]["passive"] = pair(sb[:, 0], sl[:, 0]); out["table3"]["nominalization"] = pair(sb[:, 1], sl[:, 1])

tok = GPT2TokenizerFast.from_pretrained("gpt2"); gm = GPT2LMHeadModel.from_pretrained("gpt2").to(dev).eval()
def ppl(t):
    e = tok(t, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(dev)
    with torch.no_grad():
        return math.exp(gm(e, labels=e).loss.item())
out["table3"]["perplexity"] = pair([ppl(t) for t in b.loc[ids, "summary"]], [ppl(t) for t in l.loc[ids, "summary"]])

words = lambda t: re.findall(r"[a-z0-9']+", t.lower())
def overlap(t, r, n):
    g = lambda w: [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    a, c = g(words(t)), set(g(words(r)))
    return 100 * sum(x in c for x in a) / max(len(a), 1)
for n, name in [(1, "unigram"), (2, "bigram")]:
    out["table3"][name] = pair([overlap(b.loc[i, "summary"], T[i]["reference_summary"], n) for i in ids], [overlap(l.loc[i, "summary"], T[i]["reference_summary"], n) for i in ids])

distinct = lambda t: len(set(words(t))) / max(len(words(t)), 1)
rep_b = {i for i in ids if distinct(b.loc[i, "summary"]) < .35}; rep_l = {i for i in ids if distinct(l.loc[i, "summary"]) < .35}
lw = l.loc[ids, "summary"].str.split().str.len().values; bw = b.loc[ids, "summary"].str.split().str.len().values
rw = np.array([len(r.split()) for r in refs]); dur = np.array([int(T[i]["duration"]) for i in ids]) / 60
fa = json.load(open(ROOT / "evaluation/results/error_analysis/failure_analysis.json", encoding="utf-8"))
out["counts"] = dict(videos=len(ids), r1_worse=int((l.loc[ids, "rouge1_f"] < b.loc[ids, "rouge1_f"]).sum()), repetitive_base=len(rep_b), repetitive_lora=len(rep_l),
                     words_base=bw.mean(), words_lora=lw.mean(), words_ref=rw.mean(), lora_200=int((lw >= 200).sum()),
                     within40_lora=int((np.abs(lw - rw) / rw <= .4).sum()), within40_base=int((np.abs(bw - rw) / rw <= .4).sum()),
                     framing_lora=fa["lora_statistics"]["academic_framing"], framing_base=fa["base_statistics"]["academic_framing"], citation_lora=fa["lora_statistics"]["citation_artifacts"],
                     dur_min=dur.min(), dur_max=dur.max(), dur_mean=dur.mean(), dur_median=float(np.median(dur)))
keep = [i for i in ids if i not in FAILED_ASR | rep_b | rep_l]
out["clean"] = {"n": len(keep), **{m: pair(b.loc[keep, m], l.loc[keep, m]) for m in ["rouge1_f", "rouge2_f", "rougeL_f"]}}
text = json.dumps(out, indent=1, default=float)
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(text, encoding="utf-8")
print(text)
