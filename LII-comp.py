import json
import re
import csv
from pathlib import Path
from typing import Optional, List, Tuple

from docx import Document
from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn.functional as F


# --------------------------
# CONFIG: adjust if needed
# --------------------------

LII_REF_DIR = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\LIIs")
JSON_DIR    = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\norm775")

# >>> CHANGE THIS per model
MODEL_NAME  = "sentence-transformers/LaBSE"

# >>> CHANGE THIS per model
OUTPUT_CSV  = Path(r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\LII_results_pics1to6-labse.csv")


# --------------------------
# Shared settings (keep identical across models for comparability)
# --------------------------
CHUNK_MAX_TOKENS = 128          # target max tokens INCLUDING special tokens
SPECIAL_TOKENS_BUFFER = 2       # reserve for [CLS]/[SEP]-like tokens
SHORT_TEXT_MAX_TOKENS = 128     # if fits <=128 tokens incl special => force single chunk
SHORT_TEXT_TOL = 1e-4           # warn if short-text chunk vs direct differs


# --------------------------
# Helpers
# --------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

from pathlib import Path
import re
from typing import Optional

# We compute LII only for pictures 1–6 (change if you add LII-7/LII-8 references)
VALID_PIC_IDS = set(range(1, 7))

_PIC_OR_LII_RE = re.compile(r"(?i)(?:pic|lii)\s*[-_ ]*\s*(\d+)")

def extract_picture_id_from_name(name: str) -> Optional[int]:
    """
    Extracts the numeric picture id from either:
      - ..._Pic1.json
      - LII-1.docx
    Returns int pic_id, or None if not found.
    """
    stem = Path(name).stem
    m = _PIC_OR_LII_RE.search(stem)
    if not m:
        return None
    return int(m.group(1))

def extract_subject_id_from_name(name: str) -> str:
    """
    Removes trailing _Pic# (any #) or _LII# if present.
    Examples:
      PTaherehDarbandi_Pic1.json -> PTaherehDarbandi
    """
    stem = Path(name).stem
    return re.sub(r"(?i)_(?:pic|lii)\d+$", "", stem)

def read_docx_text(path: Path) -> str:
    doc = Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return " ".join(parts)

# Persian/Arabic punctuation: ؟ ؛ plus standard . ! ?
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\u061F\u061B])\s+|\n+")

def split_into_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text]

def token_len(s: str, add_special_tokens: bool = False) -> int:
    return len(model.tokenizer(s, add_special_tokens=add_special_tokens)["input_ids"])

def chunk_sentences_to_max_tokens(sentences: List[str], max_tokens: int) -> List[str]:
    """
    Pack sentences into chunks that stay within max_tokens (including special tokens),
    using a conservative buffer for special tokens.
    """
    if not sentences:
        return []

    budget = max(8, max_tokens - SPECIAL_TOKENS_BUFFER)

    chunks: List[str] = []
    current: List[str] = []

    def current_text_plus(sent: str) -> str:
        if not current:
            return sent
        return " ".join(current + [sent])

    for sent in sentences:
        sent = normalize_text(sent)
        if not sent:
            continue

        # If a single sentence is too long, split by whitespace into smaller pieces.
        if token_len(sent, add_special_tokens=False) > budget:
            words = sent.split()
            piece: List[str] = []
            for w in words:
                cand = " ".join(piece + [w]) if piece else w
                if piece and token_len(cand, add_special_tokens=False) > budget:
                    chunks.append(" ".join(piece))
                    piece = [w]
                else:
                    piece.append(w)
            if piece:
                chunks.append(" ".join(piece))
            continue

        cand = current_text_plus(sent)
        if current and token_len(cand, add_special_tokens=False) > budget:
            chunks.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)

    if current:
        chunks.append(" ".join(current))

    return chunks

def embed_text_direct(text: str) -> torch.Tensor:
    text = normalize_text(text)
    return model.encode(text, convert_to_tensor=True, normalize_embeddings=True)

def embed_text_chunked(text: str) -> Tuple[torch.Tensor, int, int, Optional[float]]:
    """
    Returns: (doc_emb, n_tokens_no_special, n_chunks, abs_diff_vs_direct_if_short_else_None)
    """
    text = normalize_text(text)
    if not text:
        emb = embed_text_direct("")
        return emb, 0, 1, None

    n_no_special = token_len(text, add_special_tokens=False)
    n_with_special = token_len(text, add_special_tokens=True)

    # KEY FUNCTIONALITY YOU REQUESTED:
    # If the transcript fits within <=128 tokens (incl special), force a single chunk.
    if n_with_special <= SHORT_TEXT_MAX_TOKENS:
        emb_direct = embed_text_direct(text)
        emb_chunked = embed_text_direct(text)  # identical by design (single-chunk)
        # You can measure numerical drift here (should be 0.0)
        abs_diff = float(torch.max(torch.abs(emb_direct - emb_chunked)).item())
        return emb_chunked, n_no_special, 1, abs_diff

    sents = split_into_sentences(text)
    chunks = chunk_sentences_to_max_tokens(sents, CHUNK_MAX_TOKENS)
    if not chunks:
        chunks = [text]

    chunk_embs = model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)
    doc_emb = F.normalize(chunk_embs.mean(dim=0), p=2, dim=0)
    return doc_emb, n_no_special, len(chunks), None

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    return util.cos_sim(a.unsqueeze(0), b.unsqueeze(0))[0][0].item()


# --------------------------
# MAIN
# --------------------------
# ============================================================
# MODEL QUALITY / REPORTING NOTE (does not change computations)
# ============================================================
# Hugging Face explicitly flags 'sentence-transformers/bert-base-nli-mean-tokens'
# as DEPRECATED and "produces sentence embeddings of low quality".
# Use it only for *sensitivity/replication* (e.g., to mirror Bayat/Rezaii’s LII),
# not as the primary model for "best-performing" claims.
#
# SBERT docs list multilingual semantic similarity models such as:
# - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# - sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# and note LaBSE is mainly for bitext mining/translation pairs.

def print_embedding_model_notice(model_name: str) -> None:
    m = (model_name or "").strip()
    m_lower = m.lower()

    print(f"[Embedding model] {m}")

    # Flag deprecated SBERT baseline used in Bayat/Rezaii paper
    if "bert-base-nli-mean-tokens" in m_lower:
        print(
            "⚠️ MODEL QUALITY NOTE:\n"
            "  'bert-base-nli-mean-tokens' is flagged as DEPRECATED and described as producing\n"
            "  low-quality sentence embeddings on its model card.\n"
            "  Recommended use here: sensitivity/replication ONLY (to compare against Bayat/Rezaii).\n"
            "  Do NOT use it to justify the 'best' embedding backbone.\n"
            "  For semantic similarity, prefer multilingual paraphrase models such as:\n"
            "    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2\n"
            "    - sentence-transformers/paraphrase-multilingual-mpnet-base-v2\n"
        )


print(f"Loading sentence-transformer model: {MODEL_NAME}")
print_embedding_model_notice(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
model.eval()

# 1) Reference embeddings
ref_texts = {}
ref_embeddings = {}

print(f"Scanning reference folder: {LII_REF_DIR}")
for docx_path in LII_REF_DIR.glob("*.docx"):
    pic_id = extract_picture_id_from_name(docx_path.name)
    if pic_id is None:
        print(f"  [SKIP] No Pic# found in reference filename: {docx_path.name}")
        continue
    if not (1 <= pic_id <= 6):
        print(f"  [SKIP] Pic{pic_id} outside 1–6: {docx_path.name}")
        continue

    text = read_docx_text(docx_path)
    if not text.strip():
        print(f"  [WARN] Empty reference text: {docx_path.name}")
        continue

    ref_texts[pic_id] = text
    print(f"  Loaded reference for Pic{pic_id} from {docx_path.name}")

for pic_id, text in ref_texts.items():
    ref_emb, _, _, _ = embed_text_chunked(text)
    ref_embeddings[pic_id] = ref_emb

# 2) Transcript embeddings + LII
rows = []

print(f"Scanning JSON transcripts in: {JSON_DIR}")
for json_path in sorted(JSON_DIR.glob("*.json")):
    fname = json_path.name
    pic_id = extract_picture_id_from_name(fname)
    if pic_id is None:
        print(f"  [SKIP] No Pic# found in JSON filename: {fname}")
        continue
    if pic_id not in ref_embeddings:
        print(f"  [SKIP] No reference embedding for Pic{pic_id} (needed for {fname})")
        continue

    with json_path.open("r", encoding="utf8") as f:
        data = json.load(f)

    text = data.get("norm", "") or data.get("raw", "")
    if not (text or "").strip():
        print(f"  [WARN] Empty transcript in {fname}")
        continue

    emb, n_toks, n_chunks, short_abs_diff = embed_text_chunked(text)
    lii = cosine_similarity(emb, ref_embeddings[pic_id])

    if short_abs_diff is not None and short_abs_diff > SHORT_TEXT_TOL:
        print(f"  [WARN] Short-text chunk/direct drift in {fname}: {short_abs_diff:.6g}")

    subject_id = extract_subject_id_from_name(fname)

    rows.append({
        "filename": fname,
        "SubjectID": subject_id,
        "PictureID": pic_id,
        "LII": lii,
        "n_tokens": n_toks,
        "n_chunks": n_chunks,
        "short_abs_diff": short_abs_diff if short_abs_diff is not None else "",
    })

    print(f"  {fname}: Pic{pic_id}, LII={lii:.4f} | toks={n_toks} chunks={n_chunks}")

# 3) Write CSV
if rows:
    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
else:
    print("No LII rows computed – check file names and folders.")
