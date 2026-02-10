"""
dadma_wrapper.py  –  Stanza tokenizer/lemmatizer **with clitic merge**.

This version:
 • leaves ZWNJ handling to upstream `post_rules()` – no second pass;
 • merges tokens that Stanza split into <stem>  <clitic> again;
 • returns List[Dict(tok, lemma, pos)] exactly like before.
"""

from __future__ import annotations
from functools import lru_cache
from typing import List, Dict
import stanza, re

# ------------------------------------------------------------
# 0.  CONSTANTS
# ------------------------------------------------------------
_ZWNJ = "\u200c"
# clitic forms *after* post‑normalisation (no mi/ni prefixes etc.)
_CLITIC_RE = re.compile(rf"^({'|'.join(['شون','مون','تون','شان','مان','تان','ایم','اید','اند','ام','ات','اش','م','ت','ش'])})$")

# ------------------------------------------------------------
# 1.  STANZA pipeline (cached)
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_tok_pipeline():
    return stanza.Pipeline(
        lang="fa",
        processors="tokenize,mwt,pos,lemma",
        use_gpu=False,
        tokenize_no_ssplit=False,
        verbose=False,
    )

# ------------------------------------------------------------
# 2.  helper: merge <stem> <clitic>  –>  <stem+ZWNJ+clitic>
# ------------------------------------------------------------
def _merge_clitics(words: List[Dict]) -> List[Dict]:
    """
    Iterate over the token dicts returned by Stanza and merge a pronoun
    clitic if it was emitted as a *separate* token.

    Rule:   if current tok is a clitic AND previous tok does *not*
            already end with ZWNJ+that clitic  →  merge.
    """
    merged: List[Dict] = []
    i = 0
    while i < len(words):
        if (
            i > 0
            and _CLITIC_RE.match(words[i]["tok"])
            and not words[i - 1]["tok"].endswith(_ZWNJ + words[i]["tok"])
        ):
            prev = words[i - 1]
            clitic = words[i]["tok"]
            # build merged surface + lemma
            prev["tok"] += _ZWNJ + clitic
            # keep lemma of stem (prev) – you may prefer to strip clitic in a later pass
            merged[-1] = prev        # replace last inserted
            i += 1                   # skip clitic token
        else:
            merged.append(words[i])
            i += 1
    return merged

# ------------------------------------------------------------
# 3.  public API
# ------------------------------------------------------------
def dadma_tokenise(text: str) -> List[Dict]:
    """
    Tokenise + POS‑tag Persian text with Stanza, *then* apply the
    clitic‑merge post‑rule so the output aligns with the rest of the
    normalisation pipeline.
    """
    doc = _get_tok_pipeline()(text)
    out: List[Dict] = []
    for sent in doc.sentences:
        for w in sent.words:
            pos_tag = w.xpos or w.upos or "X"
            out.append({"tok": w.text, "lemma": w.lemma, "pos": pos_tag})

    # ONE pass over the whole sentence list is sufficient
    out = _merge_clitics(out)
    return out
