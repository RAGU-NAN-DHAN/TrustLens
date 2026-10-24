"""Text helpers shared by M3's modules: sentence spans, numbers, embeddings."""

import math
import re
import zlib

_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "no", "inc", "ltd", "co", "e.g", "i.e",
}
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "from", "by", "with", "as", "is", "was",
    "were", "are", "be", "been", "it", "its", "he", "she", "his", "her", "they", "their", "this", "that", "which",
    "who", "also", "has", "had", "have", "not", "but", "into", "then", "than", "such",
}
_FALLBACK_DIM = 512


def split_sentences(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, sentence) for each sentence, without splitting initials like 'A. P. J.'."""
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?=\s|$)", text):
        end = match.end()
        words = text[start:match.start()].split()
        last = words[-1].lower().rstrip(".") if words else ""
        if match.group() == "." and ((len(last) == 1 and last.isalpha()) or last in _ABBREVIATIONS):
            continue
        rest = text[end:].lstrip()
        if rest and not (rest[0].isupper() or rest[0].isdigit() or rest[0] in "\"'("):
            continue
        _append(spans, text, start, end)
        start = end
    _append(spans, text, start, len(text))
    return spans


def _append(spans: list[tuple[int, int, str]], text: str, start: int, end: int) -> None:
    segment = text[start:end]
    if not segment.strip():
        return
    lead = len(segment) - len(segment.lstrip())
    trail = len(segment) - len(segment.rstrip())
    s, e = start + lead, end - trail
    spans.append((s, e, text[s:e]))


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text)}


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _fallback_embed(text: str) -> list[float]:
    words = tokens(text)
    features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    vec = [0.0] * _FALLBACK_DIM
    for f in features:
        vec[zlib.crc32(f.encode()) % _FALLBACK_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def embed(texts: list[str]) -> list[list[float]]:
    """M1's shared embeddings when trustlens_m1 is installed with its model, else an offline placeholder."""
    try:
        from trustlens_m1.embeddings import embed_texts
        import sentence_transformers  # noqa: F401  (M1 loads it lazily; check it exists first)
    except ImportError:
        return [_fallback_embed(t) for t in texts]
    return [list(map(float, row)) for row in embed_texts(texts)]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
