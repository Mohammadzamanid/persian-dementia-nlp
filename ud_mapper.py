
# -*- coding: utf-8 -*-
"""
ud_mapper.py — compact UD-mapper for Mo's Farsi pipeline

Drop this file next to your pipeline (importable as a module).
It expects token dicts with at least keys: "tok", "lemma", "pos".
It adds/normalizes: "pos" (strip ,EZ), "feats" (dict), "misc" (dict).

Design choices:
- We keep UD UPOS in `pos` and move Ezafe to morphological features.
- Because UD Persian (Seraji/PerDT) typically does not render Ezafe as a written token,
  we record Ezafe primarily in MISC: Ezafe=Yes, and (optionally) Case=Ez.
  See: UD W20 note that ezafe is not visible in Persian Seraji (hence not marked).
- Postpositions like «پشت/بالا/کنار…» are retagged ADP when immediately followed by an
  Ezafe-marked nominal (e.g., «بالا ظرفِ ...»).

Usage:
    from ud_mapper import map_sentence_to_ud
    sent = map_sentence_to_ud(list_of_token_dicts)

Author: (spec prepared by ChatGPT for Mo)
"""

from __future__ import annotations
from typing import List, Dict

# A compact lexicon of Persian nominal/adpositional heads that function as
# postpositions when followed by an Ezafe-marked complement.
POSTPOSITION_LEXEMES: set[str] = {
    "پشت","کنار","جلو","عقب","وسط","بالا","پایین","سر","روبروی","نزدیک","پیش","داخل","کناره"
}

def _ensure_kv(obj):
    """Coerce feats/misc to dicts if they arrive as None/str."""
    if obj is None:
        return {}
    if isinstance(obj, str) and obj.strip():
        d = {}
        for part in obj.split("|"):
            if "=" in part:
                k,v = part.split("=",1)
                d[k.strip()] = v.strip()
        return d
    return dict(obj)

def map_token_to_ud(tok: Dict) -> Dict:
    """Return a *new* token dict with UD-compliant fields added/normalized."""
    t = dict(tok)  # shallow copy
    pos_raw = t.get("pos","")
    parts = [p.strip() for p in pos_raw.split(",") if p.strip()]
    has_ez = "EZ" in parts
    core_parts = [p for p in parts if p != "EZ"]
    core_pos = core_parts[0] if core_parts else pos_raw or "X"

    feats = _ensure_kv(t.get("feats"))
    misc  = _ensure_kv(t.get("misc"))

    # Add Ezafe to features/MISC without inventing a surface token
    if has_ez:
        # Language-specific value allowed under UD; keep minimal and reversible.
        feats.setdefault("Case", "Ez")
        misc["Ezafe"] = "Yes"

    t["pos"] = core_pos        # UPOS (single tag)
    t["feats"] = feats         # dict
    t["misc"]  = misc          # dict
    t["_had_ez"] = has_ez      # internal breadcrumb for downstream heuristics
    return t

def _is_ezafe_marked_nominal(tok: Dict) -> bool:
    """Heuristic: a nominal immediately carrying Ezafe (after mapping)."""
    if tok.get("pos") in {"NOUN","PROPN","ADJ","PRON","DET"}:
        ez = tok.get("misc",{}).get("Ezafe") == "Yes" or tok.get("feats",{}).get("Case") == "Ez"
        return bool(ez)
    return False

def retag_postpositions(seq: List[Dict]) -> List[Dict]:
    """
    Retag lexical nouns/adverbs like «پشت/بالا/کنار…» to ADP when they
    take an Ezafe-marked nominal complement immediately to the right.
    """
    out = [dict(t) for t in seq]
    for i, t in enumerate(out[:-1]):
        if t.get("tok") in POSTPOSITION_LEXEMES and t.get("pos") in {"NOUN","ADV","PROPN"}:
            nxt = out[i+1]
            if _is_ezafe_marked_nominal(nxt):
                t["pos"] = "ADP"
                # Optional: annotate subtype for analysis/debug
                t.setdefault("misc", {})["Retagged"] = "PostPos"
    return out

def map_sentence_to_ud(tokens: List[Dict]) -> List[Dict]:
    """
    Apply token-level mapping then contextual retagging for postpositions.
    """
    stage1 = [map_token_to_ud(t) for t in tokens]
    stage2 = retag_postpositions(stage1)
    return stage2
