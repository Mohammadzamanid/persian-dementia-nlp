"""
Compute Language Informativeness Index (LII) for picture description transcripts.

Definition (as used in Bayat/Rezaii-style LII work):
    LII = cosine_similarity( Embedding(transcript), Embedding(reference_text_for_same_picture) )

This script is part of a 3-model *sensitivity / replication* set. All three scripts share:
  • identical chunking logic (token-aware, max CHUNK_MAX_TOKENS per chunk)
  • identical embedding aggregation (mean of chunk embeddings, L2-normalized)
  • identical similarity computation (cosine similarity)
  • identical CSV schema (so outputs are directly comparable)

Why chunking?
  Sentence-Transformers truncates inputs longer than `model.max_seq_length` to the first tokens (no exception),
  so chunking is required when transcripts exceed the model context window.

Model note (important when comparing models):
  The model "sentence-transformers/bert-base-nli-mean-tokens" is explicitly marked deprecated and noted to
  produce low-quality sentence embeddings in its official model card. Use it for replication/sensitivity
  only—not for a "best model" claim. Prefer multilingual paraphrase similarity models (e.g., MiniLM/MPNet)
  for within-language semantic similarity.

References / docs:
  • SentenceTransformers max sequence length → truncation behavior.
  • LaBSE is optimized for translation / bitext mining; it can be weaker for non-translation similarity.

Author: <your name>
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from docx import Document
from sentence_transformers import SentenceTransformer, util

# --------------------------
# CONFIG (edit these paths)
# --------------------------

LII_REF_DIR = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\LIIs")
JSON_DIR = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\norm775")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_CSV = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\LII_results_pics1to6-minilm.csv")

# Pictures to score (Bayat-style picture description typically uses a fixed set).
# Your current study uses pics 1–6 and treats 7–8 as comics (no references).
VALID_PIC_IDS = set(range(1, 7))

# --------------------------
# Chunking / comparability settings
# --------------------------

# Fixed across models for comparability.
CHUNK_MAX_TOKENS = 128

# Buffer for special tokens ([CLS], [SEP], etc.). Conservative for BERT-like models.
SPECIAL_TOKENS_BUFFER = 2

# Short-text sanity check:
# If transcript fits within <= CHUNK_MAX_TOKENS (incl special tokens),
# chunking should produce a single chunk and match direct LII within tolerance.
SHORT_TEXT_TOKEN_THRESHOLD = CHUNK_MAX_TOKENS
SHORT_TEXT_ABS_DIFF_TOL = 1e-3

# Sentence splitting: English + Persian punctuation + newline boundaries
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?؟؛])\s+|\n+")

# Filename parsing:
# JSON convention:  <SubjectID>_Pic<k>.json
_JSON_PIC_AT_END_RE = re.compile(r"(?i)_pic(\d+)$")
# General (also supports reference naming: LII-1.docx / LLI-5.docx)
_ANY_PIC_RE = re.compile(r"(?i)(?:pic|lii|lli)\s*[-_ ]*\s*(\d+)")


# --------------------------
# Logging
# --------------------------

LOG_FORMAT = "%(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("LII")


# --------------------------
# Helpers: text + filename parsing
# --------------------------

def clean_text(text: str) -> str:
    """Whitespace-normalize; safe for Persian + English."""
    return re.sub(r"\s+", " ", (text or "")).strip()

def read_docx_text(path: Path) -> str:
    """Read all paragraphs from a .docx and join them."""
    doc = Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)

def extract_picture_id_from_name(name: str) -> Optional[int]:
    """
    Extract picture id from filename stem.

    Supports:
      - JSON:  PName_Pic1.json
      - DOCX:  LII-1.docx, LII_2.docx, LLI-5.docx (typo-tolerant)
    """
    stem = Path(name).stem
    m = _ANY_PIC_RE.search(stem)
    return int(m.group(1)) if m else None

def extract_subject_id_from_name(name: str) -> str:
    """Strip trailing _Pic# / _LII# / _LLI# from JSON stems."""
    stem = Path(name).stem
    return re.sub(r"(?i)_(?:pic|lii|lli)\d+$", "", stem)

def expected_pic_from_json_filename(name: str) -> Optional[int]:
    """
    Strict expectation for your JSON naming convention: *_Pic<k>.json.
    Returns the expected k if present at the end of the stem, else None.
    """
    stem = Path(name).stem
    m = _JSON_PIC_AT_END_RE.search(stem)
    return int(m.group(1)) if m else None


# --------------------------
# Helpers: chunking + embeddings
# --------------------------

def split_into_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text]

def token_len(tokenizer, text: str, add_special_tokens: bool) -> int:
    """
    Token length without truncation.

    We explicitly set tokenizer.model_max_length high to avoid HuggingFace warnings
    when *counting* tokens on long text. Actual encoding is controlled by CHUNK_MAX_TOKENS
    and SentenceTransformer's max_seq_length.
    """
    text = clean_text(text)
    if not text:
        return 0

    # Avoid "Token indices sequence length is longer than the specified maximum..."
    # This is only for counting tokens; encoding uses chunking + max_seq_length.
    if hasattr(tokenizer, "model_max_length"):
        if tokenizer.model_max_length and tokenizer.model_max_length < 10**6:
            tokenizer.model_max_length = 10**9

    ids = tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)["input_ids"]
    return len(ids)

def chunk_text_to_max_tokens(tokenizer, text: str, max_tokens: int) -> List[str]:
    """
    Return chunks of `text` such that each chunk fits within `max_tokens` (including special tokens).

    Strategy:
      1) If full text fits, return [full_text] (guarantees short-text equivalence).
      2) Else split into sentences and pack into chunks up to token budget.
      3) If a single sentence is too long, split it by whitespace.
    """
    text = clean_text(text)
    if not text:
        return []

    # Force *exactly* one chunk when short enough (<= max_tokens incl special).
    if token_len(tokenizer, text, add_special_tokens=True) <= max_tokens:
        return [text]

    budget = max(8, max_tokens - SPECIAL_TOKENS_BUFFER)
    sentences = split_into_sentences(text)

    chunks: List[str] = []
    current: List[str] = []

    def flush():
        nonlocal current
        if current:
            chunks.append(" ".join(current).strip())
            current = []

    for sent in sentences:
        sent = clean_text(sent)
        if not sent:
            continue

        # If one sentence exceeds budget, split by whitespace.
        if token_len(tokenizer, sent, add_special_tokens=False) > budget:
            words = sent.split()
            piece: List[str] = []
            for w in words:
                cand = " ".join(piece + [w]) if piece else w
                if piece and token_len(tokenizer, cand, add_special_tokens=False) > budget:
                    chunks.append(" ".join(piece).strip())
                    piece = [w]
                else:
                    piece.append(w)
            if piece:
                chunks.append(" ".join(piece).strip())
            continue

        # Try to append sentence to current chunk.
        cand = (" ".join(current + [sent])).strip() if current else sent
        if current and token_len(tokenizer, cand, add_special_tokens=False) > budget:
            flush()
            current = [sent]
        else:
            current.append(sent)

    flush()

    # Safety: never return empty
    return chunks if chunks else [text]

def embed_text_direct(model: SentenceTransformer, text: str) -> torch.Tensor:
    """Single-pass embedding (will be truncated if text > model.max_seq_length)."""
    text = clean_text(text)
    emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
    return emb

def embed_text_chunked(model: SentenceTransformer, text: str) -> Tuple[torch.Tensor, int, int, int]:
    """
    Chunk-aware embedding.

    Returns:
      doc_emb, n_tokens_no_special, n_tokens_with_special, n_chunks
    """
    text = clean_text(text)
    tok = model.tokenizer

    n_no = token_len(tok, text, add_special_tokens=False)
    n_with = token_len(tok, text, add_special_tokens=True)

    chunks = chunk_text_to_max_tokens(tok, text, CHUNK_MAX_TOKENS)
    if not chunks:
        # empty text → encode empty (rare)
        chunks = [""]

    # Hard assertion: ensure we never encode > CHUNK_MAX_TOKENS (incl special)
    for i, ch in enumerate(chunks, start=1):
        ch_len = token_len(tok, ch, add_special_tokens=True)
        if ch_len > CHUNK_MAX_TOKENS:
            raise RuntimeError(
                f"Chunk {i}/{len(chunks)} still exceeds CHUNK_MAX_TOKENS "
                f"({ch_len} > {CHUNK_MAX_TOKENS}). Please inspect chunking."
            )

    chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
    doc_emb = F.normalize(chunk_embs.mean(dim=0), p=2, dim=0)
    return doc_emb, n_no, n_with, len(chunks)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(util.cos_sim(a.unsqueeze(0), b.unsqueeze(0))[0][0].item())


# --------------------------
# Main routine
# --------------------------

def main() -> None:
    log.info(f"Model: {MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)
    model.eval()

    # Enforce identical max_seq_length across models for comparability.
    # Note: Sentence-Transformers will truncate inputs longer than this length.
    model.max_seq_length = CHUNK_MAX_TOKENS

    # Load reference embeddings per picture
    if not LII_REF_DIR.exists():
        raise FileNotFoundError(f"LII_REF_DIR not found: {LII_REF_DIR}")
    if not JSON_DIR.exists():
        raise FileNotFoundError(f"JSON_DIR not found: {JSON_DIR}")

    ref_embeddings: Dict[int, torch.Tensor] = {}
    ref_files = sorted(LII_REF_DIR.glob("*.docx"))

    log.info(f"Scanning reference folder: {LII_REF_DIR} ({len(ref_files)} .docx)")
    for docx_path in ref_files:
        pic_id = extract_picture_id_from_name(docx_path.name)
        if pic_id is None:
            log.warning(f"[SKIP] No Pic/LII id found in reference filename: {docx_path.name}")
            continue
        if pic_id not in VALID_PIC_IDS:
            log.info(f"[SKIP] Reference {docx_path.name} is Pic{pic_id}, outside {sorted(VALID_PIC_IDS)}")
            continue
        if pic_id in ref_embeddings:
            raise ValueError(f"Duplicate reference for Pic{pic_id}: {docx_path.name}")

        text = read_docx_text(docx_path)
        text = clean_text(text)
        if not text:
            log.warning(f"[SKIP] Empty reference text: {docx_path.name}")
            continue

        emb, n_no, n_with, n_chunks = embed_text_chunked(model, text)
        ref_embeddings[pic_id] = emb
        log.info(f"Loaded reference Pic{pic_id}: {docx_path.name} | toks={n_with} chunks={n_chunks}")

    missing_refs = sorted(pid for pid in VALID_PIC_IDS if pid not in ref_embeddings)
    if missing_refs:
        raise RuntimeError(
            f"Missing reference docx for pictures: {missing_refs}. "
            f"Check filenames in {LII_REF_DIR} (expected e.g., LII-1.docx ... LII-6.docx)."
        )

    # Iterate JSON transcripts
    json_files = sorted(JSON_DIR.glob("*.json"))
    log.info(f"Scanning JSON transcripts: {JSON_DIR} ({len(json_files)} .json)")

    rows: List[Dict[str, object]] = []

    for json_path in json_files:
        fname = json_path.name

        pic_id = extract_picture_id_from_name(fname)
        if pic_id is None:
            log.warning(f"[SKIP] No Pic/LII id found in JSON filename: {fname}")
            continue

        # Strict naming convention check (catches accidental bugs)
        expected = expected_pic_from_json_filename(fname)
        if expected is not None and expected != pic_id:
            raise RuntimeError(
                f"PictureID mismatch in {fname}: expected Pic{expected} from filename, "
                f"but extracted Pic{pic_id}. Check filename parsing."
            )

        if pic_id not in VALID_PIC_IDS:
            # You currently do not score Pic7/Pic8 because there are no references.
            continue

        if pic_id not in ref_embeddings:
            # Should not happen if references are complete, but keep safety.
            log.warning(f"[SKIP] No reference embedding for Pic{pic_id} (needed for {fname})")
            continue

        with json_path.open("r", encoding="utf8") as f:
            data = json.load(f)

        # Your normalized transcript field names (keep fallback)
        text = clean_text(data.get("norm", "") or data.get("raw", ""))
        if not text:
            log.warning(f"[SKIP] Empty transcript in {fname}")
            continue

        emb, n_no, n_with, n_chunks = embed_text_chunked(model, text)
        lii_chunked = cosine_similarity(emb, ref_embeddings[pic_id])

        # Short-text sanity check (only when transcript fits without chunking).
        lii_direct = ""
        abs_diff = ""
        if n_with <= SHORT_TEXT_TOKEN_THRESHOLD:
            emb_direct = embed_text_direct(model, text)
            lii_direct_val = cosine_similarity(emb_direct, ref_embeddings[pic_id])
            abs_diff_val = abs(lii_chunked - lii_direct_val)

            lii_direct = float(lii_direct_val)
            abs_diff = float(abs_diff_val)

            if n_chunks != 1 or abs_diff_val > SHORT_TEXT_ABS_DIFF_TOL:
                log.warning(
                    f"[WARN] Short-text chunking check failed for {fname}: "
                    f"n_chunks={n_chunks}, abs_diff={abs_diff_val:.6f}"
                )

        subject_id = extract_subject_id_from_name(fname)

        rows.append(
            {
                "filename": fname,
                "SubjectID": subject_id,
                "PictureID": int(pic_id),
                "LII": float(lii_chunked),
                "n_tokens_no_special": int(n_no),
                "n_tokens_with_special": int(n_with),
                "n_chunks": int(n_chunks),
                "LII_direct_if_short": lii_direct,
                "abs_diff_if_short": abs_diff,
            }
        )

    if not rows:
        raise RuntimeError("No LII rows computed. Check directories, filenames, and reference availability.")

    # Write trial-level CSV (one row per subject × picture)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "SubjectID",
        "PictureID",
        "LII",
        "n_tokens_no_special",
        "n_tokens_with_special",
        "n_chunks",
        "LII_direct_if_short",
        "abs_diff_if_short",
    ]
    with OUTPUT_CSV.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {len(rows)} rows → {OUTPUT_CSV}")

    # Write subject-level aggregation (mean LII across pictures)
    subj_vals: Dict[str, List[float]] = defaultdict(list)
    subj_pic_count: Dict[str, int] = defaultdict(int)
    for r in rows:
        subj = str(r["SubjectID"])
        subj_vals[subj].append(float(r["LII"]))
        subj_pic_count[subj] += 1

    subject_csv = OUTPUT_CSV.parent / f"{OUTPUT_CSV.stem}-subjectmean.csv"
    subj_rows: List[Dict[str, object]] = []
    for subj in sorted(subj_vals.keys()):
        vals = subj_vals[subj]
        mean = sum(vals) / len(vals)
        # population SD (ddof=0) for descriptive reporting; adjust if you prefer sample SD.
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        sd = var ** 0.5
        subj_rows.append(
            {
                "SubjectID": subj,
                "LII_mean": mean,
                "LII_sd": sd,
                "n_pictures": subj_pic_count[subj],
            }
        )

    with subject_csv.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["SubjectID", "LII_mean", "LII_sd", "n_pictures"])
        writer.writeheader()
        writer.writerows(subj_rows)
    log.info(f"Wrote subject means → {subject_csv}")

    # Write a metadata sidecar (versions + config) for reproducibility
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "device": device,
        "sentence_transformers_version": getattr(__import__("sentence_transformers"), "__version__", "unknown"),
        "torch_version": torch.__version__,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "chunk_max_tokens": CHUNK_MAX_TOKENS,
        "special_tokens_buffer": SPECIAL_TOKENS_BUFFER,
        "short_text_token_threshold": SHORT_TEXT_TOKEN_THRESHOLD,
        "short_text_abs_diff_tol": SHORT_TEXT_ABS_DIFF_TOL,
        "valid_pic_ids": sorted(VALID_PIC_IDS),
        "lii_ref_dir": str(LII_REF_DIR),
        "json_dir": str(JSON_DIR),
        "output_csv": str(OUTPUT_CSV),
        "output_subject_csv": str(subject_csv),
    }
    meta_path = OUTPUT_CSV.parent / f"{OUTPUT_CSV.stem}.meta.json"
    with meta_path.open("w", encoding="utf8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"Wrote run metadata → {meta_path}")

    log.info("Done.")


if __name__ == "__main__":
    main()
