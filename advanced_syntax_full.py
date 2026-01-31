"""
advanced_syntax_full.py

Advanced Farsi speech features using:

  • YOUR pipeline's tokenization, POS, lemmas, morphological feats (incl. Ezafe)
  • Stanza's UD Persian dependency parser (depparse_pretagged=True) 
  • A multilingual SentenceTransformer model (paraphrase-multilingual-MiniLM-L12-v2) for coherence 

Per JSON file (picture description) we compute:

  1. Dependency-based syntactic complexity
     - Mean / max / sum dependency distance
     - Dependency distance per token
     - Proportion of “long” dependencies (>= LONG_DEP_THRESHOLD)
     - Max tree depth
     - Average branching factor

  2. Clause & idea metrics (UD-based approximation)
     - Total clauses (root + clausal relations)
     - Clauses per sentence
     - Approximate "idea units" (verb + clause heads + core arguments)
     - Idea density per 10 words
     - Idea units per clause

  3. Embedding-based local coherence
     - Cosine similarity between adjacent sentence embeddings
       (mean / std / minimum)

We intentionally do NOT implement Yngve / Frazier scores, because they
require phrase-structure (constituency) trees and a validated Persian
constituency parser, which are not available as a stable, referenced
Python library.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import stanza
from stanza.models.common.doc import Document

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------
# CONFIG – EDIT THESE PATHS
# ---------------------------------------------------

INPUT_DIR = r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\norm775"
OUTPUT_CSV = r"C:\Users\Fatiima\Desktop\voices\thesis_2\advanced_syntax_full.csv"

# tokens with pos == "PUNCT" and text in this set are treated as sentence-final
END_PUNCT = {".", "؟", "!", "؛"}

# UD relations approximating clause heads
CLAUSE_RELS = {
    "root",        # main clause
    "ccomp",       # clausal complement
    "csubj",       # clausal subject
    "csubj:pass",
    "advcl",       # adverbial clause
    "xcomp",
    "acl",         # clausal modifier of noun
    "acl:relcl",
    "parataxis",   # often separate clause-like unit
}

# core argument relations (DEPID-style idea units)
ARG_RELS = {"nsubj", "nsubj:pass", "obj", "iobj"}

# Long dependency threshold, in tokens
LONG_DEP_THRESHOLD = 5


# ---------------------------------------------------
# INITIALIZE NLP MODELS
# ---------------------------------------------------

print("[INFO] Downloading / loading Stanza Persian depparse model...")
stanza.download("fa", verbose=True)
nlp = stanza.Pipeline(
    lang="fa",
    processors="depparse",
    depparse_pretagged=True,
    use_gpu=False,
)

print("[INFO] Loading multilingual sentence-transformer for coherence...")
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# supports 50+ languages, including Persian 
sent_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ---------------------------------------------------
# HELPERS: JSON → PRETAGGED SENTENCES
# ---------------------------------------------------

def feats_to_ud_string(feats):
    """
    Convert your 'feats' field (dict or string) into UD-style "Feat=Val|..." or "_".
    """
    if feats is None:
        return "_"
    if isinstance(feats, str):
        feats = feats.strip()
        return feats if feats else "_"
    if isinstance(feats, dict):
        if not feats:
            return "_"
        items = [f"{k}={v}" for k, v in feats.items()]
        return "|".join(sorted(items))
    return "_"


def json_tokens_to_pretagged_sentences(tokens: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Convert your JSON tokens (with 'tok', 'lemma', 'pos', 'feats') into a
    Stanza-compatible pretagged document:

      [[{'id', 'text', 'lemma', 'upos', 'xpos', 'feats'}, ...], ...]

    Sentence segmentation: based on PUNCT tokens with text in END_PUNCT.
    """
    sents = []
    current = []
    tid = 1

    for t in tokens:
        text = str(t.get("tok", "")).strip()
        lemma = str(t.get("lemma", text)).strip()
        upos = str(t.get("pos", "X")).strip() or "X"
        xpos = upos  # no separate XPOS tagset; re-use UPOS
        feats_ud = feats_to_ud_string(t.get("feats"))

        if not text:
            continue

        current.append({
            "id": tid,
            "text": text,
            "lemma": lemma,
            "upos": upos,
            "xpos": xpos,
            "feats": feats_ud,
        })
        tid += 1

        if upos == "PUNCT" and text in END_PUNCT:
            if current:
                sents.append(current)
            current = []
            tid = 1

    if current:
        sents.append(current)

    if not sents:
        # fallback: treat all as one sentence
        cleaned = []
        for i, t in enumerate(tokens):
            txt = str(t.get("tok", "")).strip()
            if not txt:
                continue
            cleaned.append({
                "id": i + 1,
                "text": txt,
                "lemma": str(t.get("lemma", txt)).strip(),
                "upos": str(t.get("pos", "X")).strip() or "X",
                "xpos": str(t.get("pos", "X")).strip() or "X",
                "feats": feats_to_ud_string(t.get("feats")),
            })
        if cleaned:
            sents = [cleaned]

    return sents


# ---------------------------------------------------
# DEPENDENCY-BASED SENTENCE METRICS
# ---------------------------------------------------

def dep_metrics_for_sentence(sent) -> Dict[str, float]:
    """
    Compute dependency-based metrics for a single Stanza sentence.
    """
    words = sent.words
    n = len(words)
    if n == 0:
        return {
            "dep_mean_dist": 0.0,
            "dep_max_dist": 0.0,
            "dep_sum_dist": 0.0,
            "dep_dist_per_tok": 0.0,
            "dep_prop_long": 0.0,
            "tree_max_depth": 0.0,
            "tree_mean_depth": 0.0,
            "branch_factor": 0.0,
            "clauses": 0,
            "sub_clauses": 0,
            "prop_count": 0,
        }

    dists = []
    children = defaultdict(list)
    for w in words:
        head = int(w.head) if w.head is not None else 0
        children[head].append(int(w.id))
        if head != 0:
            dists.append(abs(int(w.id) - head))

    if dists:
        dep_mean = float(np.mean(dists))
        dep_max = float(np.max(dists))
        dep_sum = float(np.sum(dists))
        dep_dist_per_tok = dep_sum / n
        dep_prop_long = sum(1 for d in dists if d >= LONG_DEP_THRESHOLD) / len(dists)
    else:
        dep_mean = dep_max = dep_sum = dep_dist_per_tok = dep_prop_long = 0.0

    # depth
    depths = {}

    def dfs(node_id: int, depth: int):
        depths[node_id] = depth
        for ch in children.get(node_id, []):
            dfs(ch, depth + 1)

    roots = [int(w.id) for w in words if int(w.head) == 0]
    for r in roots:
        dfs(r, 1)  # depth=1 at root

    if depths:
        tree_max = float(max(depths.values()))
        tree_mean = float(np.mean(list(depths.values())))
    else:
        tree_max = tree_mean = 0.0

    # branching: number of dependents per head (excluding pseudo-head 0)
    branch_counts = [len(children[h]) for h in children if h != 0]
    branch_factor = float(np.mean(branch_counts)) if branch_counts else 0.0

    # clause & simple idea count
    clauses = 0
    sub_clauses = 0
    prop_count = 0

    for w in words:
        rel = w.deprel
        upos = w.upos

        if rel in CLAUSE_RELS:
            clauses += 1
        if rel in {"ccomp", "csubj", "csubj:pass", "advcl", "xcomp", "acl", "acl:relcl"}:
            sub_clauses += 1

        # very simple DEPID-like approximation:
        #  - verbs/AUX: predicates
        #  - clause heads: event propositions
        #  - core arguments: arguments
        if upos in {"VERB", "AUX"}:
            prop_count += 1
        if rel in CLAUSE_RELS:
            prop_count += 1
        if rel in ARG_RELS:
            prop_count += 1

    return {
        "dep_mean_dist": dep_mean,
        "dep_max_dist": dep_max,
        "dep_sum_dist": dep_sum,
        "dep_dist_per_tok": dep_dist_per_tok,
        "dep_prop_long": dep_prop_long,
        "tree_max_depth": tree_max,
        "tree_mean_depth": tree_mean,
        "branch_factor": branch_factor,
        "clauses": clauses,
        "sub_clauses": sub_clauses,
        "prop_count": prop_count,
    }


# ---------------------------------------------------
# COHERENCE METRICS (EMBEDDING-BASED)
# ---------------------------------------------------

def coherence_metrics_from_tokens(sent_token_lists: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    """
    Local coherence from adjacent sentence embeddings:

      - for each sentence, join token['tok'] into a string
      - encode with multilingual MiniLM (supports Persian)
      - cosine similarity between adjacent sentence embeddings
    """
    sent_texts = []
    for sent in sent_token_lists:
        words = [str(t.get("tok", "")).strip() for t in sent if str(t.get("tok", "")).strip()]
        if not words:
            continue
        sent_texts.append(" ".join(words))

    if len(sent_texts) < 2:
        return {
            "coh_mean": 0.0,
            "coh_std": 0.0,
            "coh_min": 0.0,
        }

    embs = sent_model.encode(sent_texts)
    sims = []
    for i in range(len(embs) - 1):
        v1 = embs[i].reshape(1, -1)
        v2 = embs[i + 1].reshape(1, -1)
        sims.append(float(cosine_similarity(v1, v2)[0, 0]))
    sims = np.array(sims, dtype=float)

    return {
        "coh_mean": float(sims.mean()),
        "coh_std": float(sims.std(ddof=1)) if sims.size > 1 else 0.0,
        "coh_min": float(sims.min()),
    }


# ---------------------------------------------------
# PER-FILE PIPELINE
# ---------------------------------------------------

def process_one_json(path: str) -> Dict[str, Any]:
    """
    - read a JSON file from your pipeline
    - build pretagged Document
    - run Stanza depparse
    - compute dependency + clause + idea + coherence metrics
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    participant_id = Path(path).stem
    tokens = data.get("tokens", [])
    if not tokens:
        return {
            "Participant": participant_id,
            "Num_Sentences_dep": 0,
            "Num_Tokens_NoPunct_dep": 0,
            "Dep_MeanDist": 0.0,
            "Dep_MaxDist": 0.0,
            "Dep_SumDist": 0.0,
            "Dep_DistPerTok": 0.0,
            "Dep_PropLong": 0.0,
            "Tree_MaxDepth": 0.0,
            "Tree_MeanDepth": 0.0,
            "Branch_Factor": 0.0,
            "Clauses_Total": 0.0,
            "Clauses_per_Sent": 0.0,
            "SubClauses_per_Sent": 0.0,
            "Idea_Units": 0.0,
            "Idea_Density_per_10w_dep": 0.0,
            "Idea_per_Clause_dep": 0.0,
            "Coherence_Mean": 0.0,
            "Coherence_Std": 0.0,
            "Coherence_Min": 0.0,
        }

    # 1) Presegmented sentences from your tokens
    token_sents = []
    current = []
    for t in tokens:
        current.append(t)
        if t.get("pos") == "PUNCT" and str(t.get("tok", "")) in END_PUNCT:
            if current:
                token_sents.append(current)
            current = []
    if current:
        token_sents.append(current)
    if not token_sents:
        token_sents = [tokens]

    # 2) Build pretagged stanza document
    pretagged = json_tokens_to_pretagged_sentences(tokens)
    doc = Document(pretagged)
    doc = nlp(doc)  # runs depparse only, with your tags

    # 3) Dependency metrics aggregated
    sent_ms = []
    total_tokens_no_punct = 0

    for sent in doc.sentences:
        words = sent.words
        total_tokens_no_punct += sum(1 for w in words if w.upos != "PUNCT")
        m = dep_metrics_for_sentence(sent)
        sent_ms.append(m)

    if not sent_ms:
        dep_mean_dist = dep_max_dist = dep_sum_dist = dep_dist_per_tok = dep_prop_long = 0.0
        tree_max_depth = tree_mean_depth = branch_factor = 0.0
        clauses_total = sub_clauses_total = 0
        idea_units = 0
    else:
        dep_mean_dist = float(np.mean([m["dep_mean_dist"] for m in sent_ms]))
        dep_max_dist = float(np.max([m["dep_max_dist"] for m in sent_ms]))
        dep_sum_dist = float(np.sum([m["dep_sum_dist"] for m in sent_ms]))
        dep_dist_per_tok = float(np.mean([m["dep_dist_per_tok"] for m in sent_ms]))
        dep_prop_long = float(np.mean([m["dep_prop_long"] for m in sent_ms]))
        tree_max_depth = float(np.max([m["tree_max_depth"] for m in sent_ms]))
        tree_mean_depth = float(np.mean([m["tree_mean_depth"] for m in sent_ms]))
        branch_factor = float(np.mean([m["branch_factor"] for m in sent_ms]))
        clauses_total = int(sum(m["clauses"] for m in sent_ms))
        sub_clauses_total = int(sum(m["sub_clauses"] for m in sent_ms))
        idea_units = int(sum(m["prop_count"] for m in sent_ms))

    num_sents_dep = len(doc.sentences)
    if num_sents_dep == 0:
        clauses_per_sent = 0.0
        sub_clauses_per_sent = 0.0
    else:
        clauses_per_sent = clauses_total / num_sents_dep
        sub_clauses_per_sent = sub_clauses_total / num_sents_dep

    if total_tokens_no_punct > 0:
        idea_density_per_10w = (idea_units / total_tokens_no_punct) * 10.0
    else:
        idea_density_per_10w = 0.0

    if clauses_total > 0:
        idea_per_clause = idea_units / clauses_total
    else:
        idea_per_clause = 0.0

    # 4) Coherence metrics from your sentence texts
    coh = coherence_metrics_from_tokens(token_sents)

    return {
        "Participant": participant_id,
        "Num_Sentences_dep": float(num_sents_dep),
        "Num_Tokens_NoPunct_dep": float(total_tokens_no_punct),
        "Dep_MeanDist": dep_mean_dist,
        "Dep_MaxDist": dep_max_dist,
        "Dep_SumDist": dep_sum_dist,
        "Dep_DistPerTok": dep_dist_per_tok,
        "Dep_PropLong": dep_prop_long,
        "Tree_MaxDepth": tree_max_depth,
        "Tree_MeanDepth": tree_mean_depth,
        "Branch_Factor": branch_factor,
        "Clauses_Total": float(clauses_total),
        "Clauses_per_Sent": float(clauses_per_sent),
        "SubClauses_per_Sent": float(sub_clauses_per_sent),
        "Idea_Units": float(idea_units),
        "Idea_Density_per_10w_dep": float(idea_density_per_10w),
        "Idea_per_Clause_dep": float(idea_per_clause),
        "Coherence_Mean": coh["coh_mean"],
        "Coherence_Std": coh["coh_std"],
        "Coherence_Min": coh["coh_min"],
    }


def main():
    json_files = [
        os.path.join(INPUT_DIR, fn)
        for fn in os.listdir(INPUT_DIR)
        if fn.lower().endswith(".json")
    ]

    if not json_files:
        print(f"[ERROR] No JSON files found in {INPUT_DIR}")
        return

    rows = []
    for i, path in enumerate(sorted(json_files)):
        print(f"[INFO] {i+1}/{len(json_files)}: {os.path.basename(path)}")
        try:
            row = process_one_json(path)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}")

    if not rows:
        print("[ERROR] No features computed.")
        return

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("[OK] Saved advanced dependency + coherence features to:")
    print(f"     {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
