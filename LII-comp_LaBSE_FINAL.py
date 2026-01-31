"""
Compute Language Informativeness Index (LII) as cosine similarity between
participant transcript embeddings and picture-specific reference embeddings.

This script:
  • Uses strict Pic1–Pic6 parsing (requires 'Pic#' in filenames).
  • Applies token-aware chunking to avoid silent truncation when texts exceed the model context window.
  • Enforces a fixed chunk size (CHUNK_MAX_TOKENS=128) for comparability across embedding models.
  • Performs a short-text sanity check: for transcripts that fit within 128 tokens, chunked LII should match
    direct (non-chunked) LII within a small tolerance.

Notes:
  • Sentence-Transformers truncates any input longer than model.max_seq_length to the first tokens, without error.
    Chunking is therefore necessary if your transcripts can exceed the context length. See SBERT docs.
  • For reproducibility, we normalize embeddings and use cosine similarity.

"""

import json
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from docx import Document
from sentence_transformers import SentenceTransformer, util


# --------------------------
# CONFIG: adjust if needed
# --------------------------

# Folder with your reference texts (Farsi descriptions) as Word files
LII_REF_DIR = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\LIIs")

# Folder with your JSON transcripts (one file per subject × picture)
JSON_DIR = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\norm775")

# Where to save the output CSV
OUTPUT_CSV = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\LII_results_pics1to6-labse.csv")

# Sentence embedding model
MODEL_NAME = "sentence-transformers/LaBSE"


# --------------------------
# Chunking / comparability settings
# --------------------------

# Use the same chunk size across *all* models so the outputs are comparable.
CHUNK_MAX_TOKENS = 128

# Safety buffer for special tokens ([CLS], [SEP]) etc.
SPECIAL_TOKENS_BUFFER = 2

# Short-text equivalence check threshold (in tokens *including* special tokens)
SHORT_TEXT_TOKEN_THRESHOLD = CHUNK_MAX_TOKENS

# If short text is chunked correctly (one chunk), LII_chunked ~= LII_direct.
SHORT_TEXT_ABS_DIFF_TOL = 1e-3

# Sentence splitting: English + Persian punctuation + newline boundaries
_SENT_SPLIT_RE = re.compile(r'(?<=[\.\!\?؟؛])\s+|\n+')


# --------------------------
# Helpers: IO + filename parsing
# --------------------------

def read_docx_text(path: Path) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

def extract_picture_id_from_name(name: str) -> Optional[int]:
    """
    Accepts both conventions:
      - JSON: *_Pic1.json, *_Pic7.json
      - DOCX: LII-1.docx, LII_2.docx, (also tolerates LLI-5/LLI-6 typos)
    """
    stem = Path(name).stem
    m = re.search(r"(?i)(?:pic|lii|lli)\s*[-_ ]*\s*(\d+)", stem)
    return int(m.group(1)) if m else None

def extract_subject_id_from_name(name: str) -> str:
    """
    Strips trailing _Pic# (any number), or _LII#/_LLI# if present.
    """
    stem = Path(name).stem
    return re.sub(r"(?i)_(?:pic|lii|lli)\d+$", "", stem)



# --------------------------
# Helpers: chunking + embeddings
# --------------------------

def split_into_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text]

def token_len(text: str, add_special_tokens: bool = False) -> int:
    return len(model.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

def chunk_text_to_max_tokens(text: str, max_tokens: int) -> List[str]:
    """
    Return a list of chunks, each small enough to encode without truncation.

    Strategy:
      1) If full text already fits, return [full_text] (guarantees short-text equivalence).
      2) Else split into sentences and pack them into chunks up to the token budget.
      3) If a single sentence is too long, split it by words.

    We budget max_tokens - SPECIAL_TOKENS_BUFFER for token_len(..., add_special_tokens=False).
    """
    text = clean_text(text)
    if not text:
        return []

    # If it fits (including special tokens), we force exactly one chunk.
    if token_len(text, add_special_tokens=True) <= max_tokens:
        return [text]

    budget = max(8, max_tokens - SPECIAL_TOKENS_BUFFER)

    sentences = split_into_sentences(text)
    chunks: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(" ".join(current).strip())
            current = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # If a sentence itself is too long, split it by words.
        if token_len(sent, add_special_tokens=False) > budget:
            flush_current()
            words = sent.split()
            piece: List[str] = []
            for w in words:
                cand = (" ".join(piece + [w])).strip()
                if piece and token_len(cand, add_special_tokens=False) > budget:
                    chunks.append(" ".join(piece))
                    piece = [w]
                else:
                    piece.append(w)
            if piece:
                chunks.append(" ".join(piece))
            continue

        # Try adding to current chunk
        cand = (" ".join(current + [sent])).strip() if current else sent
        if current and token_len(cand, add_special_tokens=False) > budget:
            flush_current()
            current = [sent]
        else:
            current.append(sent)

    flush_current()

    # Last resort: should not happen, but avoid returning empty list
    return chunks if chunks else [text]

def embed_text_direct(text: str) -> torch.Tensor:
    """
    Single-pass encoding (may truncate if text > model.max_seq_length).
    """
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

def embed_text_chunked(text: str) -> Tuple[torch.Tensor, int, int, int]:
    """
    Chunk-aware encoding that avoids truncation:
      • Encodes each chunk separately (normalized).
      • Averages chunk embeddings.
      • Re-normalizes the averaged vector.

    Returns:
      (doc_embedding, n_tokens_no_special, n_tokens_with_special, n_chunks)
    """
    text = clean_text(text)
    if not text:
        raise ValueError("Empty text")

    n_no_special = token_len(text, add_special_tokens=False)
    n_with_special = token_len(text, add_special_tokens=True)

    chunks = chunk_text_to_max_tokens(text, CHUNK_MAX_TOKENS)
    chunk_embs = model.encode(
        chunks,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    doc_emb = F.normalize(chunk_embs.mean(dim=0), p=2, dim=0)
    return doc_emb, n_no_special, n_with_special, len(chunks)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return util.cos_sim(a.unsqueeze(0), b.unsqueeze(0))[0][0].item()


# --------------------------
# MAIN
# --------------------------

print(f"Loading sentence-transformer model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# For comparability + safety, enforce the same max_seq_length across all scripts.
# Longer inputs would be truncated by Sentence-Transformers (hence our chunking).
try:
    model.max_seq_length = min(int(model.max_seq_length), CHUNK_MAX_TOKENS)
except Exception:
    model.max_seq_length = CHUNK_MAX_TOKENS


# --------------------------
# 1. Load reference texts & build embeddings per picture
# --------------------------
VALID_PIC_IDS = set(range(1, 7))
ref_embeddings: Dict[int, torch.Tensor] = {}

print(f"Scanning reference folder: {LII_REF_DIR}")
for docx_path in sorted(LII_REF_DIR.glob("*.docx")):
    pic_id = extract_picture_id_from_name(docx_path.name)
    if pic_id is None:
        print(f"  [SKIP] No Pic/LII id found in reference filename: {docx_path.name}")
        continue
    text = read_docx_text(docx_path)
    if pic_id not in VALID_PIC_IDS:
        print(f"  [SKIP] Reference {docx_path.name} is Pic{pic_id}, outside {sorted(VALID_PIC_IDS)}")
        continue



    emb, n_no, n_with, n_chunks = embed_text_chunked(text)
    ref_embeddings[pic_id] = emb
    print(f"  Loaded reference for Pic{pic_id} ({n_with} toks incl. special, {n_chunks} chunk(s)) from {docx_path.name}")

if not ref_embeddings:
    raise RuntimeError(f"No reference embeddings were built. Check LII_REF_DIR: {LII_REF_DIR}")


# --------------------------
# 2. Iterate over JSON transcripts, compute LII
# --------------------------

rows: List[Dict[str, object]] = []

print(f"Scanning JSON transcripts in: {JSON_DIR}")
for json_path in sorted(JSON_DIR.glob("*.json")):
    fname = json_path.name

    pic_id = extract_picture_id_from_name(fname)
    if pic_id is None:
        print(f"  [SKIP] No 'Pic#' found in JSON filename: {fname}")
        continue

    if pic_id not in ref_embeddings:
        print(f"  [SKIP] No reference embedding for Pic{pic_id} (needed for {fname})")
        continue

    with json_path.open("r", encoding="utf8") as f:
        data = json.load(f)

    # Your normalized Farsi transcript field names (keep your original fallback)
    text = data.get("norm", "") or data.get("raw", "")
    text = clean_text(text)
    if not text:
        print(f"  [WARN] Empty transcript in {fname}")
        continue

    emb, n_no, n_with, n_chunks = embed_text_chunked(text)
    lii_chunked = cosine_similarity(emb, ref_embeddings[pic_id])

    # Short-text equivalence check
    lii_direct = ""
    abs_diff = ""
    if n_with <= SHORT_TEXT_TOKEN_THRESHOLD:
        emb_direct = embed_text_direct(text)
        lii_direct_val = cosine_similarity(emb_direct, ref_embeddings[pic_id])
        abs_diff_val = abs(lii_chunked - lii_direct_val)

        lii_direct = float(lii_direct_val)
        abs_diff = float(abs_diff_val)

        if n_chunks != 1 or abs_diff_val > SHORT_TEXT_ABS_DIFF_TOL:
            print(
                f"  [WARN] Short-text chunking check failed for {fname}: "
                f"n_chunks={n_chunks}, abs_diff={abs_diff_val:.6f}"
            )

    subject_id = extract_subject_id_from_name(fname)

    rows.append({
        "filename": fname,
        "SubjectID": subject_id,
        "PictureID": pic_id,
        "LII": float(lii_chunked),
        "n_tokens_no_special": int(n_no),
        "n_tokens_with_special": int(n_with),
        "n_chunks": int(n_chunks),
        "LII_direct_if_short": lii_direct,
        "abs_diff_if_short": abs_diff,
    })

    print(f"  {fname}: Pic{pic_id}, LII={lii_chunked:.4f}, toks={n_with}, chunks={n_chunks}")


# --------------------------
# 3. Write CSV
# --------------------------

if rows:
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

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
else:
    print("No LII rows computed – check file names and folders.")
