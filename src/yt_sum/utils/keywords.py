# keywords.py
from __future__ import annotations
from typing import List, Tuple, Optional
import re
import math

# Optional deps
try:
    from keybert import KeyBERT
    _HAS_KEYBERT = True
except Exception:
    _HAS_KEYBERT = False

try:
    import yake
    _HAS_YAKE = True
except Exception:
    _HAS_YAKE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SK = True
except Exception:
    _HAS_SK = False

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

_NUMBER = re.compile(r"""
    (?<![A-Za-z0-9])
    (?:\d{1,3}(?:,\d{3})*|\d+)
    (?:\.\d+)?      
    (?:\s*(?:%|ppm|ms|s|m|h|Hz|kHz|MHz|GHz|kbps|Mbps|Gbps|dB|V|mV|A|mA|W|kW|N|Pa|bar|°C|K|mol|g|kg|mg|µg|mm|cm|m|km))?
    (?![A-Za-z0-9])
""", re.X)

_ACRO = re.compile(r'\b[A-Z][A-Z0-9]{1,7}\b')
_FORMULA = re.compile(r'(\$[^$]+\$|\\\(.*?\\\)|\\\[.*?\\\]|[A-Za-z0-9\)\]]\s*=\s*[A-Za-z0-9\(\[][^\.]{0,80})')


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return _SENT_SPLIT.split(text.strip())


def extract_critical_terms(text: str, top_k_nums: int = 30) -> List[str]:
    terms = []
    terms += [m.group(0) for m in _NUMBER.finditer(text)]
    terms += [m.group(0) for m in _ACRO.finditer(text)]
    terms += [m.group(0) for m in _FORMULA.finditer(text)]
    seen = set()
    kept = []
    for t in terms:
        if t not in seen:
            kept.append(t)
            seen.add(t)
    return kept[:max(10, top_k_nums)]


def extract_keywords(text: str, top_n: int = 12, method: str = "auto", max_ngram: int = 3) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if method in ("auto", "keybert") and _HAS_KEYBERT:
        try:
            kw = KeyBERT()
            pairs = kw.extract_keywords(text, keyphrase_ngram_range=(1, max_ngram), stop_words="english", top_n=top_n)
            return [p[0] for p in pairs]
        except Exception:
            pass

    if method in ("auto", "yake") and _HAS_YAKE:
        try:
            extractor = yake.KeywordExtractor(lan="en", n=max_ngram, top=top_n)
            candidates = extractor.extract_keywords(text)
            candidates.sort(key=lambda x: x[1])
            return [c[0] for c in candidates[:top_n]]
        except Exception:
            pass

    if method in ("auto", "tfidf") and _HAS_SK:
        try:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, max_ngram), max_features=5000)
            X = vec.fit_transform([text])
            scores = X.toarray()[0]
            terms = vec.get_feature_names_out()
            pairs = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
            return [p[0] for p in pairs[:top_n]]
        except Exception:
            pass

    words = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text.lower())
    stop = set("""
        the a an of to and in on at for from by with as is are was were be been being
        that this those these it its into out about we you they them he she his her
        or nor but so if then than too very can may might should would could will
    """.split())
    freq = {}
    for w in words:
        if w in stop:
            continue
        freq[w] = freq.get(w, 0) + 1
    pairs = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:top_n]]


def select_evidence_sentences(transcript: str, keywords: List[str], top_k: int = 12, diversity: float = 0.35) -> List[str]:
    sents = split_sentences(transcript)
    if not sents:
        return []
    if not keywords:
        step = max(1, len(sents)//top_k)
        return [sents[i].strip() for i in range(0, len(sents), step)[:top_k]]

    kw_ranks = {kw.lower(): (len(keywords) - i) for i, kw in enumerate(keywords)}
    scored = []
    for s in sents:
        s_low = s.lower()
        score = 0.0
        for kw, rk in kw_ranks.items():
            if kw in s_low:
                score += 1.0 + 0.1 * rk
        if score > 0:
            scored.append((score, s.strip()))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return []

    selected: List[str] = []
    selected_vecs: List[set] = []
    def sent_vec(text: str) -> set:
        return set(re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text.lower()))
    for score, sent in scored:
        if len(selected) >= top_k:
            break
        sv = sent_vec(sent)
        penalty = 0.0
        for vec in selected_vecs:
            inter = len(sv & vec)
            denom = math.sqrt(len(sv)+1) * math.sqrt(len(vec)+1)
            sim = inter/denom if denom else 0.0
            penalty = max(penalty, sim)
        final = score - diversity * penalty * score
        if final > 0:
            selected.append(sent)
            selected_vecs.append(sv)
    return selected


def highlight_sentences(transcript: str, keywords: List[str], top_k: int = 6, diversity: float = 0.35) -> List[str]:
    ev = select_evidence_sentences(transcript, keywords, top_k=max(top_k, 6), diversity=diversity)
    return ev[:top_k]
