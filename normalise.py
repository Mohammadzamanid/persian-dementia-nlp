#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
normalise.py – unified Persian (Farsi) normalization + UD-oriented postprocess

This pipeline implements:
  • Pre-tokenization canonicalization + phrase/token replacements (ZWNJ hygiene)
  • Robust tokenization via spaCy/Dadma & clitic/mi merges
  • Contextual retagging for closed-class words, clitics, LVCs, and PP-like items
  • Manual POS & lemma overrides
  • Ezafe postprocessing (blocklist, predicative guard, pairwise suppressions)
  • Light-verb construction (LVC) repairs (POS + Ezafe cleanup)
  • Conservative noun-lemma fallback

Changes marked “AUDIT FIX:” directly address recurring errors reported in
batch1/batch2 (e.g., بعد=ADV, برای not Ezafe, بالا→بالای + Ezafe, stray «ش»,
حیوان‌ات from incorrect ZWNJ, LVC نشان می‌دهد, interjection «آخ…»). See the
audit files for examples.  """

from __future__ import annotations
import argparse
import json
import re
import string
import sys
from pathlib import Path
from importlib import import_module, util, machinery

# =============================================================================
# 1) Dynamic imports (keep project layout-agnostic)
# =============================================================================

_ZWNJ = "\u200c"

def _dynamic_import(module_name: str, hint_file: str | None = None):
    """Attempt package import; fall back to local file if running as script."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError:
        if hint_file is None:
            raise
        here = Path(__file__).resolve()
        for parent in here.parents:
            cand = parent / hint_file
            if cand.is_file():
                spec = util.spec_from_loader(
                    module_name + "_dyn",
                    machinery.SourceFileLoader(module_name, str(cand))
                )
                mod = util.module_from_spec(spec)
                sys.modules[spec.name] = mod
                spec.loader.exec_module(mod)
                return mod
        raise

# External helpers (local fallbacks permitted)
from .char_canon    import canon_chars              # canonical chars/ZWNJ policy
from .hazm_wrapper  import hazm_normalise           # optional text normaliser
from .spacy_wrapper import spacy_tokenise as dadma_tokenise  # spaCy/Dadma

try:
    from .ud_mapper import map_sentence_to_ud
except Exception:
    _ud_mod = _dynamic_import("ud_mapper", "ud_mapper.py")
    map_sentence_to_ud = getattr(_ud_mod, "map_sentence_to_ud")

# =============================================================================
# 2) Load manual maps & replacements
# =============================================================================

def _canon_key(s: str) -> str:
    s = canon_chars(s)
    s = s.replace(_ZWNJ, "").replace("-", "")
    return re.sub(r"\s+", "", s)

def _canon_val_nozwnj(v: str) -> str:
    return canon_chars(v).replace(_ZWNJ, "")

try:
    _lemma_mod = _dynamic_import("manual_maps", "manual_maps.py")
    _RAW_VERB_MAP: dict[str, str]  = getattr(_lemma_mod, "manual_lemma_map", {})
    _RAW_NOUN_MAP: dict[str, str]  = getattr(_lemma_mod, "noun_manual_lemma_map", {})
    _MANUAL_POS_MAP: dict[str, str] = getattr(_lemma_mod, "manual_pos_map", {})
except Exception:
    _RAW_VERB_MAP, _RAW_NOUN_MAP, _MANUAL_POS_MAP = {}, {}, {}

try:
    _norm_mod = _dynamic_import("replacements", "replacements.py")
    _RAW_REPLACEMENTS: dict[str, str] = getattr(_norm_mod, "replacements", {})
except Exception:
    _RAW_REPLACEMENTS = {}

if not _RAW_REPLACEMENTS or not _RAW_VERB_MAP:
    raise RuntimeError("Ensure 'replacements.py' and 'manual_maps.py' are accessible.")

VERB_MAP: dict[str, str] = {_canon_key(k): _canon_val_nozwnj(v) for k, v in _RAW_VERB_MAP.items()}
NOUN_MAP: dict[str, str] = {_canon_key(k): canon_chars(v) for k, v in _RAW_NOUN_MAP.items()}

# Split replacements into single-token vs multi-token/phrase entries
TOKEN_REPLACEMENTS: dict[str, str] = {
    _canon_key(k): canon_chars(v)
    for k, v in _RAW_REPLACEMENTS.items()
    if not re.search(r'[\s\u200c\-]', k)
    
}
_PHRASE_ENTRIES: list[tuple[str, str]] = [
    (k, canon_chars(v))
    for k, v in _RAW_REPLACEMENTS.items()
    if re.search(r'[\s\u200c\-]', k)
]
TOKEN_REPLACEMENTS[_canon_key("باهم")] = canon_chars("با هم")


# =============================================================================
# 3) Lexicon & regex defs
# =============================================================================

# Words whose final sequences may resemble clitic suffixes but are lexical (no clitic).
# AUDIT FIX: prevent false ZWNJ insertion in Arabic (-ات) plurals like «حیوانات/تشکیلات».
NON_CLITIC_WORDS = {"هم", "درخت", "پشت","برگشت", "آب‌وتاب","منتها","ناراحت","یک‌خورده","نشان","کشان","وقت","دولت","ملت","رمضان","صنعت","تجارت","عدالت","سعادت","شهادت","طبیعت","شریعت","حقیقت","معرفت",
    "مصلحت","حکمت","رحمت","نعمت","برکت","وحدت","نسبت","دعوت","حکومت","قیمت","هویت","شهرت","لیوان","گلدان","نشان","میوه","میز","خانواده","تحت","میزگرد",'حکایت',
    "ثروت","قیامت","بصیرت","غیرت","محتویات","حیرت","حسرت","لغت","صحت","سلامت","طراوت","صحبت","تربیت","لیوان","سیگار","کاشان","ایران","کرمان","اصفهان",'رفت','پنکه','گوله','چمباتمه','ریزش',
    "محبت","فرصت","رسالت","خوش","فروشان","مراسم","جام","چیست","چی","آفرینش","موش","امانت","حمایت","شکایت","روایت","دست","نباید","انواع و اقسام","انواع‌واقسام","فنجان",'نیاید','بیاید','نی‌اید','بی‌اید','شاید','باید','ایشان','ساختمان','ساختمون','اقلام','همان','داستان‌سرایی','اعلام','سمت','راست','نظافت','کریم‌خان','حوضچه','همه',
     "حیوون","باید","حیوان","شباهت","سرگرم","سیم","هجوم","حیات","برگی","مهمان","دخترخانوم","دوش","نقش","معلوم","دخترخانم","فرش","فروش","فرش‌فروش","تصمیم","لوازم","گرم","درویش","دوم","سوم","چهارم","انتهاش","آسمان","قدیم","اهرم","سلام","آدم","کوهستان","انجام","هندوانه","ایناهاش","مردم","خدمت","مامان","قلم","علم","ظلم","رحم","ختم","رقم","حکم","ستم","فهم","قدم","قسم","کرم","اسم",
    "اینجاش","جسم","رسم","فلش","مهم","کمان","شهرستان","حرکات","مرحوم","دستفروش","بلم","رفتها","سیمان","رفت‌ها","میوه‌جات","میوهجات","اممم","چشم","حالات","آلات","نظم","آسمون","کفش","دبستان","چهارم","دوران","تمام","قایم","آب‌کش","نجات","قفسه","خاطرات","لرزش","نت","دنبال","شکلات","اعدام","داستان","جزئیات","احترام","اقدام","اهتمام","فیلم","تیم","حرم",'دم',"پنجره","کوچه","گوجه",
     "بال‌اش","باباش","نیست‌ها","محکم","خش","بسم","الرحیم","لغزش","دوش","مس‌فروش","مسفروش","روکش","پیش","درختان","هرکدوم","بعدش","هرکدام","جیم","یواش‌یواش","یواشیواش","نیستها","تنقلات","می‌کنندها","میکنندها","هیچ‌کدوم","پیش","گردش","بیش","علیکم","هیچکدوم","کش","نوجوان","جدی","پرت","بالاش","اونجاش","آمم","آینه","ریش","آتش","دانش","کوشش","ریش‌ریش","باید","سفره","آرایش","نمایش","حرکت","فرش","شیرینی‌جات","بخش","چکش","نقاش","فروش","خروش","خواهش","تلاش",    "میوه","کوه","توهُم","شانه","خانه","پنجره","قصه","قصابه","کرخه","کوزه",
    "آرامش","نوازش","ترکش","کدام","تیروکمان","مستقیم","تابش","دوستان","تموم","ورزش","خاموش","تنها","رفت‌ها","رفتها","مریم","جهان","لم","تصمیمات","کاهش","آرام","حالات","افزایش","کاپیتالیسم","سوسیالیسم","گرایش","بینش","کنش","اقسام","واکنش","بنفش","آغوش","پرچم","حالت","گلیم","هوش","نقوش","گوش","دوردست","عدم","درست","است","هست","نیست","است.", "هست.", "نیست."
    "عطش","پوشش","آموزش","پژوهش","داداش","گلاب‌پاش","پرسش","گزارش","سفارش", "میز","میوه","ساعت","بایست","قسمت","ن‌شان","تلویزیون","پخش","جهت","کابینت","خانم",'سیستم',
    "حیوانات","تشکیلات","مشکلات","امم","مشخصات","نکات","صفات","اشتباهات","علایم","ایام","زمان","دیوان","باهم", "علائم","دستورات","هم","آها","نوشابه‌فروش","نوشابهفروش", "انتها",'غزلیات','کم','نردبان','اعم',}

_NONCL_CANON = {_canon_key(w) for w in NON_CLITIC_WORDS}

def _is_nonclitic_lexeme(s: str) -> bool:
    return _canon_key(s) in _NONCL_CANON

_PUNCT = string.punctuation + "؟،؛«»…"
_SUFFIX_PRON  = r'(?:ام|ات|اش|ایم|اید|اند|مان|تان|شان|مون|تون|شون|م|ت|ش)'
_SUFFIX_BLOCK = rf'(?:ها(?:ی)?(?:{_SUFFIX_PRON})?|{_SUFFIX_PRON})'

# =============================================================================
# 4) Pre-tokenization helpers
# =============================================================================

def _flex_pat_from_phrase(key_raw: str) -> re.Pattern:
    """A tolerant pattern for phrase replacements (spaces/ZWNJ/hyphen fold)."""
    parts = re.split(r'[\s\u200c\-]+', canon_chars(key_raw).strip())
    sep   = r'(?:[\s\u200c\-]+)'
    core  = sep.join(map(re.escape, parts))
    return re.compile(rf'(?<![^\W\d_]){core}(?![^\W\d_])', flags=re.UNICODE)

_PHRASE_REGEXES: list[tuple[re.Pattern, str]] = [
    
    (_flex_pat_from_phrase(k), v)
    for k, v in sorted(_PHRASE_ENTRIES, key=lambda kv: len(canon_chars(kv[0])), reverse=True)
    
]

# --- AUDIT HARD-PATCH: phrase overrides for this batch ---
# 1) Preserve lexical ZWNJ compound «عارف‌مسلک»
_PHRASE_REGEXES.insert(0, (_flex_pat_from_phrase("عارف مسلک"), "عارف\u200cمسلک"))


# 2) Disambiguate colloquial «توکار مس» → preposition + noun «تو کار مس»
#    (extremely surgical; only fires on the exact bigram «توکار مس»)
_PHRASE_REGEXES.insert(0, (_flex_pat_from_phrase("توکار مس"), "تو کار مس"))

def _apply_phrase_replacements(text: str) -> str:
    if not _PHRASE_REGEXES:
        return text
    for rx, rep in _PHRASE_REGEXES:
        text = rx.sub(rep, text)
    return text

def _apply_token_replacements(text: str) -> str:
    """Token-wise replacements using hazm WordTokenizer if available."""
    if not TOKEN_REPLACEMENTS:
        return text
    try:
        from hazm import WordTokenizer
        tokens = WordTokenizer().tokenize(text)
    except Exception:
        tokens = text.split()

    out = []
    for tok in tokens:
        lead, core, tail = "", tok, ""
        while core and core[0] in _PUNCT: lead += core[0]; core = core[1:]
        while core and core[-1] in _PUNCT: tail = core[-1] + tail; core = core[:-1]
        if core:
            core = TOKEN_REPLACEMENTS.get(_canon_key(core), core)
        out.append(lead + core + tail)
    return " ".join(out)

#def _insert_zwnj_before_suffix_block(text: str) -> str:
#    """
#    Insert ZWNJ before plural/pronominal suffix blocks when they are *really* suffixes.
#    AUDIT FIX: Skip when the *whole* word is a known lexical item (e.g., «حیوانات»).
#    """
#    #pattern = re.compile(rf'\b([^\s\u200c]+?)({_SUFFIX_BLOCK})(?!\u200c)\b')
#    #def replacer(m):
#        #base, suff = m.group(1), m.group(2)
#        #full = base + suff
#        # If the full form is a canonical lexeme (e.g., Arabic plural -ات), do nothing.
#        #if _is_nonclitic_lexeme(full):
#            #return full
#        # Otherwise inject ZWNJ before suffix block
#        #return base + _ZWNJ + suff
#    #return pattern.sub(replacer, text)

def _undo_false_zwnj_on_lexemes(tokens):
    for t in tokens:
        s = t.get("tok","")
        if "\u200c" in s and _is_nonclitic_lexeme(s.replace("\u200c","")):
            t["tok"] = s.replace("\u200c","")
    return tokens

def _strip_illegal_zwnj_before_pron(text: str) -> str:
    pron_suf = r'(?:ام|ات|اش|ایم|اید|اند|مان|تان|شان|مون|تون|شون|م|ت|ش)\b'
    return re.sub(rf'(?<!ه)\u200c(?={pron_suf})', '', text)

def _post_rules(text: str) -> str:
    # Normalize (ن)می + stem → enforce single ZWNJ; fix residual whitespace
    text = re.sub(r'\b(ن?می)\s+([^\s]+)\b', r'\1' + _ZWNJ + r'\2', text)

    _mi_bad = {_canon_key(w) for w in NON_CLITIC_WORDS if w.startswith("می")}
    def _join_mi(m):
        pref, rest = m.group(1), m.group(2)
        cand = _canon_key(pref + rest)
        base = re.sub(r'(?:ها|های|ان)?(?:ام|ات|اش|مان|تان|شان|مون|تون|شون|م|ت|ش)?$', '', cand)
        # do NOT join if the whole word is a known lexical item (e.g., میوه/میز/…)
        if cand in _mi_bad or base in _mi_bad:
            return m.group(0)
        return pref + _ZWNJ + rest

    text = re.sub(r'\b(ن?می)([^\s\u200c]{3,})\b', _join_mi, text)
        # h + plural/pronominal suffix → h+ZWNJ+suffix
    text = re.sub(r'(ه)(?=(?:ها|های)\b)', r'\1' + _ZWNJ, text)
    text = re.sub(r'(ه)(?=(?:ام|ات|اش|مان|تان|شان)\b)', r'\1' + _ZWNJ, text)
    text = re.sub(r'([^\s\d‌\u200c]+تر)ه\b', r'\1 است', text)
    text = re.sub(r'(\S*ه)(?:\u200c)?ان\b', r'\1‌اند', text)
    text = re.sub(r'([اآبپتثجچحخدذرزسشصضطظعغفقکگلمنوهی]+تر)ه\b', r'\1 است', text)

    text = _strip_illegal_zwnj_before_pron(text)
    text = re.sub(r'(?<=\S)\u200c(?=ی\b)', '', text)
    return text

def _pretoken_colloquial(text: str) -> str:
    text = re.sub(r'(?<=\S)شو(?=(?:\s|[؟\.,،؛!»\)\]]|$))', 'ش را', text)
    # 1) 3pl colloquial copula: «…ه‌ان» → «…ه‌اند»
    text = re.sub(r'(\S*?ه)\u200c?ان\b', r'\1‌اند', text)
    # 2) Predicative comparative/adj: «…تره» → «…تر است»
    text = re.sub(r'(\S*?تر)\s*ه\b', r'\1 است', text)
    # 3) Generic predicative after these preheads: «این/آن/خیلی/بسیار … ه» → «… است»
    return text
# =============================================================================
# 5) Post-tokenization helpers
# =============================================================================

def _apply_manual_pos_overrides(tokens: list[dict]) -> list[dict]:
    """Hard override POS for known systemic cases (from audit + error logs)."""
    for t in tokens:
        form = t.get("tok", "")
        if form in _MANUAL_POS_MAP:
            t["pos"] = _MANUAL_POS_MAP[form]
    return tokens


# --- plural / indefinite endings (compile once) ---
PLURAL_RE = re.compile(r'(?:(?<!ه)\u200c?ها|ه\u200cها)(?:\u200c?ی)?$')  # …ها / …‌ها, or …ه‌ها ; optional Ezafe ی
INDEF_RE  = re.compile(r'ه(?:\u200c)?(?:ای|ئی)$')                        # …ه‌ای / …ه‌ئی
AR_LETTER_RE = re.compile(r'[اآ-ی]')

def _ensure_clitic_flags(tokens: list[dict]) -> list[dict]:
    for t in tokens:
        s = t.get("tok") or ""
        surf_nz = canon_chars(s).replace(_ZWNJ, "")

        # Standalone «ها/های» is not a plural suffix; also not a PRON host.
        if surf_nz in {"ها", "های","تنها","تنهاتنها"}:
            t["had_plural_suffix"] = False
            t["had_pron_clitic"] = False
            t["had_yfinal"] = s.endswith("ی")
            continue

        if surf_nz in {"فروشان","فرش‌فروشان","فرشفروشان","مسگران"}:
            t["had_plural_suffix"] = True
            continue

        # 1) Pron. clitic detection by morph segments (if any)
        has_pron_clitic = any(seg.get("role") == "PRON_CL" for seg in (t.get("morph_segments") or []))
        t["had_pron_clitic"] = bool(has_pron_clitic)

        # 2) If a pronoun clitic is present, peel it off before plural testing
        core_for_plural = s
        if has_pron_clitic:
            core_for_plural, _ = _match_longest_suffix(s, tuple(_PRON_MAP.keys()))

        # 3) Plural «…ها(+ی)» (robust to ZWNJ and final-heh); require a real stem before it
        m_pl = PLURAL_RE.search(core_for_plural or "")

        # guard against accidental zero-width matches (e.g., if PLURAL_RE is ever made optional)
        if m_pl and m_pl.start() == m_pl.end():
            m_pl = None
        has_stem = False
        if m_pl:
            stem_before = canon_chars(core_for_plural[:m_pl.start()]).replace(_ZWNJ, "")
            has_stem = bool(AR_LETTER_RE.search(stem_before))  # blocks bare «ها/های»
        candidate_plural = bool(m_pl and has_stem)

        # 4) Indefinite «…ه‌ای/ه‌ئی» is only relevant if we did NOT detect plural
        indef_heh_y = (not candidate_plural) and bool(INDEF_RE.search(core_for_plural or ""))

        # 5) Hard guard: lexical forms that merely *look* like clitic/plural endings
        if _is_nonclitic_lexeme(surf_nz):
            t["had_pron_clitic"] = False
            t["had_plural_suffix"] = False

        # Keep your existing explicit overrides (unchanged)
        elif t.get("tok") in {"کمدهای", "نورهایی","پشتی‌هایی", "دوستان","چادرهایی","چراغ‌های","پنجره‌های","بازیکنان", "شمعدونی‌های","شرایط‌های","ریش‌هایش", "بدهی‌های","تصمیمات","آلات","میوه‌جات","حالات","برگ‌هایش","اینجاهایی","فروشان","پرده‌هایی","بیل‌هایی","گل‌هایی","گل‌های","مس‌هایی","بیل‌های","پشتی‌های","بازرگان‌های","دکه‌های","دکه‌ها","مبل‌های","آقاهایی"}:
            t["had_plural_suffix"] = True
        elif t.get("tok") in {"گل‌هاش","پاروهایش","رنگ‌هایش","گل‌هایش","برگ‌هایش","درختان","همه‌شان","پایه‌هایش","ریش‌هایش","قسمت‌هایش",}:
            t["had_plural_suffix"] = True
            t["had_pron_clitic"] = True
        elif t.get("tok") in {"نصفش","می‌فرستتش","عینکم"}:
            t["had_pron_clitic"] = True
        elif t.get("tok") in {"عینکم"}:
            t["had_pron_clitic"] = True

        # Default: trust the robust plural detector; do NOT suppress by indef
        else:
            t["had_plural_suffix"] = candidate_plural

        # 6) Final-Y marker (orthographic)
        t["had_yfinal"] = s.endswith("ی")

    return tokens


def _retag_closed(tokens: list[dict]) -> list[dict]:
    """
    Normalize closed-class POS with priority ordering.
    AUDIT FIX: add INTJ items like «آخ/آخ‌آخ…»; ensure «چه»، copulas, etc.
    """
    _CLOSED_ALLOWED = {
        "ما":{"PRON"}, "من":{"PRON"}, "او":{"PRON"}, "بدون": {"ADP"},
        "و":{"CCONJ"}, "یا":{"CCONJ"}, "که":{"SCONJ"}, "اینکه":{"SCONJ"},
        "در":{"ADP"}, "به":{"ADP"}, "از":{"ADP"}, "تا":{"ADP"}, "با":{"ADP"},
        "داخل":{"ADP"}, "مثل":{"ADP"},
        "دیگر": {"ADV","DET","ADJ"},
        "دیگری": {"ADV","DET","ADJ"},"بدون": {"ADP"},
        "را":{"PART"},"ان‌شاءالله": {"INTJ"},"ایشالله":{"INTJ"},
        "تو":{"ADP","PRON"}, "این":{"DET","PRON"}, "آن":{"DET","PRON"}, "اون":{"DET","PRON"},
        "آها":{"INTJ"}, "آهان":{"INTJ"},"الفرار":{"INTJ"},"امم":{"INTJ"}, "آه":{"INTJ"}, "وای":{"INTJ"},"آره":{"INTJ"},"بلی":{"INTJ"},"بله":{"INTJ"},"اوه":{"INTJ"},"آخ‌آخ‌آخ‌آخ": {"INTJ"},
        # AUDIT: ensure interjections
        # AUDIT: ensure interjections
        "آخ":{"INTJ"},"اوکی":{"INTJ"}, "آخ‌آخ":{"INTJ"},"والا":{"INTJ"}, "آخ‌آخ‌آخ":{"INTJ"}, "آخ‌آخ‌آخ‌آخ":{"INTJ"}, "الحمدالله":{"INTJ"},"به‌به":{"INTJ"},"خداحافظ":{"INTJ"},"خدافظ":{"INTJ"},
        "هم":{"PART"},
        # copulas/auxes
        "باید":{"AUX"}, "نباید":{"AUX"}, "هستند":{"AUX"}, "ست":{"AUX"},
        "باشد":{"AUX"}, "بود":{"AUX"}, "بودند":{"AUX"}, "است":{"AUX"},
        "هست":{"AUX"}, "نیست":{"AUX"},
        # interrogative
        "چه":{"DET","PRON","SCONJ"}
    }
    _CLOSED_PRIORITY = ("PART","ADV","ADP","AUX","CCONJ","INTJ","DET","PRON","SCONJ")
    for t in tokens:
        surf = canon_chars(t.get("tok","").replace(_ZWNJ, ""))
        allowed = _CLOSED_ALLOWED.get(surf)
        if not allowed: 
            continue
        current = t.get("pos","")
        current_ok = any(current.startswith(a) for a in allowed)
        if not current_ok:
            # pick highest priority among allowed
            for p in _CLOSED_PRIORITY:
                if p in allowed:
                    t["pos"] = p
                    break
    return tokens

def _is_headlike_next(pos: str) -> bool:
    return any(pos.startswith(p) for p in ("NOUN","ADJ","NUM","PROPN"))

def _fix_heh_ezafe_graph(tokens: list[dict]) -> list[dict]:
    """
    If a token ends with ...هی (plain 'ه' + 'ی' with no ZWNJ) and the next token
    looks like a nominal/adjectival head, rewrite as 'ه‌ی' (ZWNJ).
    This addresses forms like «صندوقچهی» → «صندوقچه‌ی».
    """
    n = len(tokens)
    for i, t in enumerate(tokens):
        surf = t.get("tok", "")
        if len(surf) >= 2 and surf.endswith("ی") and surf[-2] == "ه":
            # find the first non‑PUNCT follower
            j = i + 1
            while j < n and tokens[j].get("pos") == "PUNCT":
                j += 1
            if j < n and tokens[j].get("pos", "").startswith(("NOUN","ADJ","PROPN","NUM")):
                # insert ZWNJ between 'ه' and 'ی'
                t["tok"] = surf[:-1] + "\u200c" + "ی"
    return tokens

def _retag_contextual(tokens: list[dict]) -> list[dict]:
    """
    Context dependent retagging.
      • دنبال → ADP before NP/VP
      • وارد → ADJ licensing complement or passive/resultative contexts
      • پای (noun) vs ADP noise
      • کنار → ADP before NP
      • یک‌دانه → NUM (not ADV)
      • تو → ADP vs PRON (peek ahead)
      • چپ/راست as ADJ in سمت/دست constructions
      • AUDIT FIX: LVC nouns (e.g., «نشان») must be NOUN before light verbs
      • AUDIT FIX: stand‑alone clitic forms (ش/م/ت/…): force PRON
    """
    n = len(tokens)
    LIGHT_VERBS = {"دادن","کردن","زدن","گرفتن","داشتن","شدن","کشیدن","داد"}
    LVC_NOUNS   = {
        "نشان","کمک","حرف","قدم","تلاش","تصمیم","صحبت","تماس","شروع","پایان",
        "خرید","فروش","بازی","اشاره","پرت","فکر","نگاه","عرض","جلب","کار","تماشا",
    }
    for i, t in enumerate(tokens):
        surf = canon_chars(t.get("tok",""))
        surf_nz = surf.replace(_ZWNJ, "")
        pos  = t.get("pos","")

        if surf_nz == "داخل" and pos == "ADP" and i > 0 and tokens[i-1].get("tok") == "می‌آید":
           t["pos"] = "ADV"
           continue

        if (t.get("tok") == "دنبال") and i > 0 and tokens[i+1].get("tok") == "گربه" and tokens[i+2].get("tok") == "می‌کند": 
            t["pos"] = "NOUN"
            continue

        if surf_nz == "دنبال" and i + 2 < len(tokens):
           nxt, nxt2 = tokens[i+1], tokens[i+2] 
           if (nxt.get("lemma") == "گربه" and nxt.get("pos","").startswith("NOUN")
               and nxt2.get("lemma") == "کردن" and nxt2.get("pos","").startswith("VERB")):
               t["pos"] = "NOUN"
               continue
           
        if t.get("tok") == "پس" and tokens[i-1].get("pos") == "PUNCT":
            t["pos"] = "ADV"
            continue


        # ADP-like follow-ups
        if surf_nz.startswith("دنبال"):
            if pos == "NOUN" and i + 1 < n and tokens[i+1].get("pos") in {"NOUN","PROPN","VERB"}:
                t["pos"] = "ADP"
            continue

        

        if surf == "وارد" and not pos.startswith("ADJ"):
            if i + 1 < n and (tokens[i+1].get("pos","").startswith(("NOUN","PROPN")) or tokens[i+1].get("lemma") in {"شدن","کردن"}):
                t["pos"] = "ADJ"
            continue

        if surf == "پای" and pos.startswith("ADP"):
            if i > 0 and tokens[i-1].get("pos") == "ADP":
                t["pos"] = "NOUN"
            continue

        if surf == "کنار" and pos.startswith("ADV"):
            if i + 1 < n and tokens[i+1].get("pos","").startswith(("NOUN","PROPN")):
                t["pos"] = "ADP"
            continue

        if surf == "دور" and pos.startswith(("ADP","ADV")):
            if i + 1 < n and tokens[i+1].get("pos","").startswith(("NOUN","PROPN","PRON")):
               t["pos"] = "NOUN"
            continue

        if surf == "یک‌دانه" and pos.startswith("ADV"):
            t["pos"] = "NUM"
            continue

        # این/آن as DET before head, else PRON
        if surf in {"این","آن","اون"}:
            j = i + 1
            while j < n and tokens[j].get("pos") in {"DET","PART","PUNCT","ADV"}:
                j += 1
            next_pos = tokens[j].get("pos","") if j < n else ""
            t["pos"] = "DET" if _is_headlike_next(next_pos) else "PRON"
            continue

        if surf in {"دیگر", "دیگری"}:
            # default (clausal) use: ADV
            new_pos = "ADV"

            # Pre-nominal: "دیگر کتاب/کار/..."  → ADJ
            if i + 1 < n and tokens[i+1].get("pos","").startswith(("NOUN","PROPN","ADJ","NUM","PRON")):
                new_pos = "ADJ"

            # Post-nominal: "کارهای دیگر/کتاب دیگر/..."  → ADJ
            if new_pos == "ADV":
                p = i - 1
                while p >= 0 and tokens[p].get("pos") == "PUNCT":
                    p -= 1
                if p >= 0 and tokens[p].get("pos","").startswith(("NOUN","PROPN","NUM","PRON")):
                    new_pos = "ADJ"

            t["pos"] = new_pos
            continue


        if surf == "خانمی":
            t["pos"] = "NOUN"
            t["lemma"] = "خانم"
            continue

        # تو as ADP before NP/ADJ/NUM/PROPN/PRON
        if surf_nz == "تو":
            j = i + 1
            while j < n and tokens[j].get("pos") in {"DET","PART","PUNCT","ADV"}:
                j += 1
            nxt_pos = tokens[j].get("pos","") if j < n else ""
            t["pos"] = "ADP" if nxt_pos.startswith(("NOUN","ADJ","NUM","PROPN","PRON")) else "PRON"
            continue


        # چپ/راست as ADJ in "سمت/دست چپ/راست"
        if surf in {"چپ","راست"} and not pos.startswith("ADJ"):
            if i > 0 and tokens[i-1].get("lemma") in {"دست","سمت"}:
                t["pos"] = "ADJ"
            continue

        # AUDIT FIX: LVC retagging — ensure noun before a light verb window
        if (t.get("lemma") in LVC_NOUNS or surf in LVC_NOUNS):
            # If currently mis-tagged as VERB, flip to NOUN when a light verb follows.
            next_is_lv = False
            for j in (i+1, i+2):
                if j < n:
                    if tokens[j].get("lemma","") in LIGHT_VERBS or tokens[j].get("pos","").startswith("VERB"):
                        next_is_lv = True
                        break
            if next_is_lv and not t.get("pos","").startswith("NOUN"):
               t["pos"] = "NOUN"
            t["no_verb_override"] = True

            if pos == "NOUN" and re.search(r'(?:دن|تن)$', surf) and surf not in {"بدن","میهن","وطن","دندان"}:
                prev_pos = tokens[i-1].get("pos","") if i > 0 else ""
                # Masdar contexts: after ADP/DET/SCONJ/CCONJ/PART or when bearing Ezafe → KEEP as NOUN
                if prev_pos in {"ADP","DET","SCONJ","CCONJ","PART"} or t.get("misc",{}).get("Ezafe") == "Yes":
                    pass
                else:
                    t["pos"] = "VERB"
                continue
        

        # AUDIT FIX: clitic tokens accidentally standalone (e.g., 'ش')
        if surf in {"ش","م","ت","مان","تان","شان","مون","تون","شون"}:
            t["pos"] = "PRON"
            continue 


        if t.get("pos","").startswith("VERB") and t.get("tok","").endswith("ی"):
            if t.get("lemma") in NOUN_MAP:
                # If it’s not immediately modifying another noun (i+1), prefer NOUN
                for j in (i+1, i+2):
                    if j < len(tokens) and _is_copula(tokens[j]):
                        t["pos"] = "NOUN"
                        break

        if pos.startswith("NOUN") and (surf.endswith("ن") or str(t.get("lemma","")).endswith("ن")):
            prev_lem = tokens[i-1].get("lemma","") if i > 0 else ""
            if prev_lem in {"مشغول", "حال"}:  # «در حال»
                t["pos"] = "VERB"

        if surf in {"جلو", "جلوی"} and (t.get("had_pron_clitic") or (i + 1 < n and tokens[i+1].get("pos") in {"NOUN", "PRON", "PROPN", "DET"})):
            t["pos"] = "ADP"
    
        if surf == "دور":
            if i + 1 < n:
                next_tok_text = tokens[i+1].get("tok", "")
                next_tok_pos = tokens[i+1].get("pos", "")
                if next_tok_text == "هم":
                    t["pos"] = "ADV"
                elif next_tok_pos in {"NOUN", "PRON", "PROPN", "DET"}:
                    t["pos"] = "ADP"

        # Progressive AUX: دارد … می‌V…
        if t.get("lemma") == "داشتن" and not t.get("pos","").startswith("AUX"):
            # Look ahead up to 4 tokens, which covers "دارد + obj1 + obj2 + می‌VERB"
            for j in range(i + 1, min(i + 5, n)):
                next_tok = tokens[j]
                # If we find the main verb, tag 'دارد' as AUX and stop searching
                if next_tok.get("pos","").startswith("VERB") and next_tok.get("tok","").startswith("می"):
                    t["pos"] = "AUX"
                    break # IMPORTANT: Exit the inner loop once found
                # If we hit a boundary (like another verb or punctuation), stop looking
                if next_tok.get("pos") in {"VERB", "AUX", "SCONJ", "PUNCT"}:
                    break

        if canon_chars(t.get("tok","")) == "برداشت" and t.get("pos","").startswith("VERB"):
            j = i + 1
            if j < n and tokens[j].get("lemma") == "من":
                k = j + 1
                if k < n and tokens[k].get("tok") == "از":
                    t["pos"] = "NOUN"
                    t["lemma"] = "برداشت"

        # B1-P2: «فضای بیرون خانه» → بیرون = NOUN
        if canon_chars(t.get("tok","")) == "بیرون" and t.get("pos") == "ADV":
            if i > 0 and tokens[i-1].get("tok") in {"فضای","فضا"}:
                if i + 1 < n and tokens[i+1].get("pos","").startswith("NOUN"):
                    t["pos"] = "NOUN"

        if (t.get("tok") == "زدن"):
            prev_tok = tokens[i-1].get("tok") if i > 0 else ""
            if prev_tok in {"چرت", "پارو"} and _has_aux_within(tokens, i, max_ahead=3):
                _force_infinitive_pos(t)       # sets VERB + VerbForm=Inf + stem morph_pos=V
                t["lemma"] = "زدن"             # keep lemma consistent with other masdars
                continue

        if (t.get("tok") == "همین") and i > 0 and tokens[i+1].get("tok") == "در" and tokens[i+2].get("tok") == "حین" and tokens[i+3].get("tok") == "بازی":
            t["pos"] = "DET"
            continue

        if (t.get("tok") == "شستن") and i > 0 and tokens[i-1].get("tok") == "ظرف" and tokens[i+1].get("pos") == "PUNCT":
            t["pos"] = "VERB"
            continue

        if (t.get("tok") == "بیشتر") and i > 0 and tokens[i-1].get("tok") == "که":
            t["pos"] = "ADV"
            continue

        if (t.get("tok") == "دارند") and i > 0 and tokens[i+1].get("tok") == "آن" and tokens[i+2].get("tok") == "یا" and tokens[i+3].get("tok") == "ساعت" and tokens[i+4].get("tok") == "را" and tokens[i+5].get("tok") == "نگاه" and tokens[i+6].get("tok") == "می‌کنند":
            t["pos"] = "AUX"
            continue

        if (t.get("tok") == "دارد") and i > 0 and tokens[i+1].get("tok") == "با" and tokens[i+2].get("tok") == "بچه" and tokens[i+3].get("tok") == "مشغول" and tokens[i+4].get("tok") == "بازی" and tokens[i+5].get("tok") == "توپ" and tokens[i+6].get("tok") == "است":
            t["pos"] = "AUX"
            continue

        if (t.get("tok") == "نشستنش") and i > 0 and tokens[i-1].get("tok") == "حالت":
            t["pos"] = "NOUN"
            continue

        if (t.get("tok") == "دارد") and i > 0 and tokens[i+1].get("tok") == "در" and tokens[i+2].get("tok") == "حال" and tokens[i+3].get("tok") == "رفتن" and tokens[i+4].get("tok") == "است": 
            t["pos"] = "AUX"
            continue

        if (t.get("tok") == "دارند") and i > 0 and tokens[i+1].get("tok") == "مشغول" and tokens[i+2].get("tok") == "تماس" and tokens[i+3].get("tok") == "تلفنی" and tokens[i+4].get("tok") == "هستند": 
            t["pos"] = "AUX"
            continue


        if canon_chars((t.get("tok") or "")).replace(_ZWNJ, "") == "بالای":
            j = _next_non_punct(tokens, i+1)
            if j is not None:
                nxt_lem = canon_chars((tokens[j].get("lemma") or tokens[j].get("tok") or "")).replace(_ZWNJ, "")
                if tokens[j].get("pos","").startswith(("NOUN","PROPN","PRON")) and nxt_lem in {"طاقچه","نردبان","درخت","توپ"}:
                    t["pos"] = "ADP"
                    t.setdefault("feats", {})["Case"] = "Ez"
                    t.setdefault("misc",  {})["Ezafe"] = "Yes"
                    continue


        if canon_chars((t.get("tok") or "")).replace(_ZWNJ, "") == "سمت" and i > 0 and tokens[i-1].get("tok") == "به":
            j = _next_non_punct(tokens, i+1)
            if j is not None and tokens[j].get("pos","").startswith(("NOUN","PROPN","PRON")):
                t["pos"] = "ADP"
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                continue

        surf_nozwnj = canon_chars(t.get("tok","")).replace("\u200c", "")
        if surf_nozwnj == "عارفمسلک" or t.get("tok") == "عارف\u200cمسلک":
            t["pos"]   = "ADJ"
            t["lemma"] = "عارف\u200cمسلک"
            continue

    return tokens

_AGR_MAP = {"م": ("1", "Sing"), "ی": ("2", "Sing"), "د": ("3", "Sing"),
            "یم": ("1", "Plur"), "ید": ("2", "Plur"), "ند": ("3", "Plur"),
            "اند": ("3", "Plur")}

_PRON_MAP = {"م": ("1", "Sing"), "ت": ("2", "Sing"),"ام": ("1","Sing"), "ات": ("2","Sing"), "ش": ("3", "Sing"),"اش": ("3", "Sing"),
             "مان": ("1", "Plur"), "تان": ("2", "Plur"), "شان": ("3", "Plur"),
             "مون": ("1", "Plur"), "تون": ("2", "Plur"), "شون": ("3", "Plur")}
def _match_longest_suffix(s: str, cand: tuple[str, ...]) -> tuple[str, str]:
    """
    Return (stem_without_suffix, suffix) matching the **longest** candidate at the end of s,
    comparison ignores ZWNJ; if none, return (s, "").
    """
    s_nz = s.replace(_ZWNJ, "")
    for suf in sorted(cand, key=len, reverse=True):
        if s_nz.endswith(suf):
            # remove suf from the non-ZWNJ string, then rebuild stem by slicing from the right
            k = len(suf)
            stem_nz = s_nz[:-k]
            # reconstruct stem from original by consuming non-ZWNJ chars to length of stem_nz
            acc = []
            count = 0
            for ch in s:
                if ch != _ZWNJ:
                    count += 1
                if count <= len(stem_nz):
                    acc.append(ch)
            stem = "".join(acc)
            return stem, s[len(stem):]  # suffix portion with original ZWNJ if any
    return s, ""

PAST_STEMS_N: tuple[str, ...] = (
    "نشست", "نوشت", "نگاشت", "نهاد", "نمود", "نواخت"
)

# --- normalized imperfective/negative prefixes ---
IMPF_PREFIXES = ("می‌", "می")
NEG_IMPF_PREFIXES = ("نمی‌", "نمی")
# NEW: safe extraction of imperfective prefix + AGR for verbs
def _extract_verb_affixes(surface: str) -> dict:
    """
    Return a dict {'prefix','neg','stem','agr'} for a Persian finite verb form.

    Design:
      • NEG only when part of «نمی‌…/نمی…» OR a bare «ن…» directly precedes a present form
        with AGR; guard against lexical past stems beginning with «نـ» (e.g., نشست-).
      • Imperfective/present (می‌…): AGR endings {م،ی،د،یم،ید،ند}.
      • Handle archaic 3SG presents {آید،گوید،گیرد} → AGR=د with stem trimmed.
      • Simple past / bare present-subjunctive (no می‌): same AGR surface endings.
      • Preserve the glide ی in the present stem of «آمدن»: «می‌آیم/می‌آید/…» → stem=«آی».

      HARD GUARD (present 3SG): treat «کند» as stem=«کن», AGR=«د», even with optional
      NEG and/or «می». This prevents the erroneous parse stem=«ک» + AGR=«ند».
    """
    import re

    s = surface or ""
    out = {"prefix": "", "neg": "", "stem": "", "agr": ""}

    # ---- HARD GUARD for present 3SG «کند» (with optional NEG and/or می) ----
    base = s
    had_neg = False
    had_impf = False
    for pre in ("نمی‌", "نمی", "می‌", "می"):
        if base.startswith(pre):
            if pre.startswith("ن"):
                had_neg = True
            if "می" in pre:
                had_impf = True
            base = base[len(pre):]
            break
    else:
        if base.startswith("ن"):  # bare negation without می
            had_neg = True
            base = base[1:]


    if base.replace("\u200c", "") == "کند":
        if had_neg:
            out["neg"] = "ن"
        if had_impf:
            out["prefix"] = "می‌"
        out["stem"] = "کن"
        out["agr"]  = "د"
        return out

    # ---- collapse ZWNJ for detection only (do NOT rewrite the original surface here) ----
    s_nz = s.replace("\u200c", "")

    # ---- 1) Prefix / NEG detection ------------------------------------------------------
    found = False

    # (a) NEG+IMPF first: «نمی‌… / نمی…»
    for neg_impf in ("نمی‌", "نمی"):
        if s.startswith(neg_impf):
            out["neg"] = "ن"
            out["prefix"] = "می‌"      # normalize to ZWNJ form in features
            s = s[len(neg_impf):]
            found = True
            break

    if not found:
        # (b) bare IMPF: «می‌… / می…»
        for impf in ("می‌", "می"):
            if s.startswith(impf):
                out["prefix"] = "می‌"
                s = s[len(impf):]
                found = True
                break

    if not found:
        # (c) optional bare NEG «ن…» before a present form (avoid lexical نـ past stems)
        if s.startswith("ن") and not s.startswith(("نمی", "نمی‌")) \
           and not any(s_nz.startswith(p) for p in PAST_STEMS_N):
            rest = s[1:]
            if re.search(r"(?:م|ی|د|یم|ید|ند)$", rest):
                out["neg"] = "ن"
                s = rest

    # ---- 2) Strip AGR endings; compute stem --------------------------------------------

    # Ultra‑conservative 3SG disambiguation for ambiguous «…ند» surfaces attested in data.
    # Keep this list TINY to avoid regressions; add entries only when seen in gold data.
    def _force_3sg_nd(form: str) -> bool:
        # «بیند» (از «دیدن») → stem=«بین» + agr=«د»
        return form == "بیند"

    stem = s
    agr = ""

    if out["prefix"]:
        # Imperfective / present
        if s in {"آید", "گوید", "گیرد"}:            # archaic 3SG presents
            stem, agr = s[:-1], "د"
        elif s.endswith("یم"):
            stem, agr = s[:-2], "یم"
        elif s.endswith("ید"):
            stem, agr = s[:-2], "ید"
        elif s.endswith("اند"):
            if re.search(r'ه[\u200c]?اند$', s):
                stem, agr = s[:-3], "اند"            # e.g., «می‌کرده‌اند»
            else:
                stem, agr = s[:-1], "د"              # e.g., «می‌خواند»
        elif s.endswith("ند"):
            if _force_3sg_nd(s):                     # e.g., «می‌بیند» → 3SG
                stem, agr = s[:-1], "د"
            elif len(s) == 3:                        # «کند» (already guarded above)
                stem, agr = s[:-1], "د"
            elif s.endswith("نند"):                  # rare double‑n typo → still plural
                stem, agr = s[:-2], "ند"
            else:
                stem, agr = s[:-2], "ند"             # true 3PL
        elif s.endswith("م"):
            stem, agr = s[:-1], "م"
        elif s.endswith("ی"):
            stem, agr = s[:-1], "ی"
        elif s.endswith("د"):
            stem, agr = s[:-1], "د"
        else:
            stem, agr = s, ""
    else:
        # Simple past / bare present‑subjunctive
        if s.endswith("یم"):
            stem, agr = s[:-2], "یم"
        elif s.endswith("ید"):
            stem, agr = s[:-2], "ید"
        elif s.endswith("اند"):
            if re.search(r'ه[\u200c]?اند$', s):
                stem, agr = s[:-3], "اند"
            else:
                stem, agr = s[:-1], "د"
        elif s.endswith("ند"):
            if _force_3sg_nd(s):                     # bare «بیند» → 3SG
                stem, agr = s[:-1], "د"
            elif len(s) == 3:
                stem, agr = s[:-1], "د"
            elif s.endswith("نند"):
                stem, agr = s[:-2], "ند"
            else:
                stem, agr = s[:-2], "ند"
        elif s.endswith("م"):
            stem, agr = s[:-1], "م"
        elif s.endswith("ی"):
            stem, agr = s[:-1], "ی"
        elif s.endswith("د"):
            stem, agr = s[:-1], "د"
        else:
            stem, agr = s, ""

    # ---- 3) Normalize present stem of «آمدن»: keep glide ی ------------------------------
    if out["prefix"] and surface.startswith(("می‌آی", "نمی‌آی")) and stem and not stem.startswith("آی"):
        stem = "آی"

    out["stem"] = stem
    out["agr"] = agr
    return out



def _merge_bare_pron_tokens(tokens: list[dict]) -> list[dict]:
    """If a bare clitic (ش/م/ت/…/شان) was tokenized separately, merge it to the left token
       for non-verbal hosts."""
    PRON_TOKS = {"ش","م","ت","مان","تان","شان","مون","تون","شون","اش"}
    out = []
    for t in tokens:
        if t.get("tok") in PRON_TOKS and out and not out[-1].get("pos","").startswith("VERB"):
            out[-1]["tok"] = (out[-1]["tok"] or "") + t["tok"]
            out[-1]["had_pron_clitic"] = True
            # drop current token
            continue
        out.append(t)
    return out

def _fix_zwnj_before_pron_on_tokens(tokens: list[dict]) -> list[dict]:
    """
    Remove ZWNJ immediately before pronominal clitics unless preceded by 'ه'.
    Fixes outputs like «کنار‌ش/جلوی‌ش/بالا سر‌ش» → «کنارش/جلویش/بالا سرش».
    """
    pron = r'(?:ام|ات|اش|ایم|اید|اند|مان|تان|شان|مون|تون|شون|م|ت|ش)\b'
    bad = re.compile(rf'(?<!ه)\u200c(?={pron})')
    for t in tokens:
        if t.get("pos","").startswith("VERB"):
            continue
        s = t.get("tok","") or ""
        s = bad.sub("", s)
        s = re.sub(r'\u200c{2,}', '\u200c', s)
        t["tok"] = s
    return tokens


def _coarse_pos(p: str) -> str:
    if not p:
        return ""
    # take the first label before comma, then strip subtype after colon
    return str(p).split(",")[0].split(":")[0]

def _force_morphpos_from_pos(tokens: list[dict]) -> list[dict]:
    POS2MP = {"NOUN":"N","PROPN":"N","ADJ":"ADJ","ADV":"ADV","ADP":"ADP","PRON":"PRON",
              "PART":"PART","CCONJ":"CCONJ","SCONJ":"SCONJ","NUM":"NUM","AUX":"AUX","PUNCT":"PUNCT"}
    for t in tokens:
        base = _coarse_pos(t.get("pos",""))
        mp   = "V" if base.startswith("VERB") else POS2MP.get(base)
        if not mp:
            continue
        for seg in t.get("morph_segments", []):
            if seg.get("role") == "stem":
                seg["morph_pos"] = mp
    return tokens



def _create_morph_segments(tokens: list[dict]) -> list[dict]:
    """
    Build morph_segments per token (Two-Layer Principle) with deterministic
    disentanglement of verbal agreement (AGR) vs pronominal clitics (PRON_CL).
    """
    PRON_ENDINGS = tuple(_PRON_MAP.keys())
    PRON_FOR_VERB = tuple(
    k for k in _PRON_MAP.keys()
    if k not in {"ام", "ات", "ایم", "اید", "اند", "م"}
)

    for t in tokens:
        s = t.get("tok", "") or ""
        pos = t.get("pos", "") or ""
        s_nz = s.replace(_ZWNJ, "")
        morph: list[dict] = []

        def _push_stem(form: str, mpos: str):
            if form:
                morph.append({"form": form, "role": "stem", "morph_pos": mpos})

        # ------ VERBS & AUXILIARIES (e.g., copula) ------
        if pos.startswith(("VERB","AUX")):
            # Always exclude copular-like long endings on verbs:
            core_after_pron, pron = _match_longest_suffix(s, PRON_FOR_VERB)
            s_nz = s.replace(_ZWNJ, "")
            if pron == "م" or s_nz.endswith("یم") and pron:
                pron = ""
                core_after_pron = s

            # Accept object PRON only if the prior chars look verbal
            accept_pron = False
            if pron and not _is_nonclitic_lexeme(s_nz):
                prior = core_after_pron.replace(_ZWNJ, "")
                if pron in {"شون","شان","مون","تون","مان","تان"} and re.search(r'(?:^|[^ا-ی])(?:می\u200c?|نمی\u200c?)?[^\s]*ند$', s_nz):
                    accept_pron = False
                elif re.search(r'(?:م|ی|د|یم|ید|ند|ه)$', prior):
                    accept_pron = True

            if accept_pron:
                parts = _extract_verb_affixes(core_after_pron)
                if parts.get("neg"):
                    morph.append({"form": parts["neg"], "role": "PREF_NEG"})
                if parts.get("prefix"):
                    morph.append({"form": parts["prefix"], "role": "PREF_IMPF"})
                _push_stem(parts["stem"], "V")
                agr = parts.get("agr", "")
                if agr:
                    per, num = _AGR_MAP[agr]
                    morph.append({"form": agr, "role": "AGR", "Person": per, "Number": num})
                pkey = pron.replace(_ZWNJ, "")
                per, num = _PRON_MAP[pkey]
                morph.append({"form": pron, "role": "PRON_CL", "Person": per, "Number": num, "Case": "Obj"})
                t["had_pron_clitic"] = True
            else:
                parts = _extract_verb_affixes(s)
                if parts.get("neg"):
                    morph.append({"form": parts["neg"], "role": "PREF_NEG"})
                if parts.get("prefix"):
                    morph.append({"form": parts["prefix"], "role": "PREF_IMPF"})
                _push_stem(parts["stem"], "V")
                agr = parts.get("agr", "")
                if agr:
                    per, num = _AGR_MAP[agr]
                    morph.append({"form": agr, "role": "AGR", "Person": per, "Number": num})
                t["had_pron_clitic"] = False

        # ------ NOUN/ADP/ADJ/ADV hosts of possessive PRON_CL ------
        elif pos.startswith(("NOUN", "ADP", "ADJ", "ADV")):
            if not _is_nonclitic_lexeme(s_nz):
                core, pron = _match_longest_suffix(s, PRON_ENDINGS)
            else:
                core, pron = s, ""



            # Cancel false positives when ZWNJ is inside a compound rather than clitic boundary
            if pron:
                ends_with_zwnj_pron = s.endswith(_ZWNJ + pron)
                ends_with_heh_pron  = s.endswith("ه" + pron) or s.endswith("ه" + _ZWNJ + pron)
                    # (1) Arabic plural «ـات» ≠ 2SG poss.clitic unless «ه/‌» immediately precedes
                if pron == "ات" and not (ends_with_zwnj_pron or ends_with_heh_pron):
                    # Prefer plural reading when the singular «…ه» exists (or plausibly exists)
                    if _is_nonclitic_lexeme((core or "").replace(_ZWNJ, "") + "ه"):
                        core, pron = s, ""  # keep whole token as plural noun

                # (2) Bare «ـت» ≠ 2SG poss.clitic unless «ه/‌» immediately precedes
                if pron == "ت" and not (ends_with_zwnj_pron or ends_with_heh_pron):
                    core, pron = s, ""
                    
                if _ZWNJ in s and not (ends_with_zwnj_pron or ends_with_heh_pron):
                    last_z = s.rfind(_ZWNJ)
                    pron_start = len(s) - len(pron)
                    if not (0 <= last_z < pron_start - 0):
                        core, pron = s, ""

            base_pos = ("N" if pos.startswith("NOUN")
                        else "ADP" if pos.startswith("ADP")
                        else "ADJ" if pos.startswith("ADJ")
                        else "ADV")

            # Early guard: never split lexical plurals or items in NON_CLITIC_WORDS
            if _is_nonclitic_lexeme(s_nz):
                _push_stem(s, base_pos)
                t["had_pron_clitic"] = False
            else:
                if pron:
                    pkey = pron.replace(_ZWNJ, "")
                    per, num = _PRON_MAP[pkey]

                    # Peel an immediately-preceding plural block:  … + «ها» [«ی» optional] + (pron)
                    m = re.search(r'(?:\u200c)?ها(?:(?:\u200c)?ی)?$', core or "")
                    if m:
                        base = (core or "")[:m.start()]
                        # Push bare stem, then an explicit plural suffix segment
                        _push_stem(base, base_pos)
                        morph.append({"form": "ها", "role": "SUFF_PL"})
                        t["had_plural_suffix"] = True  # idempotent with prior flag
                    else:
                        _push_stem(core, base_pos)

                    morph.append({"form": pron, "role": "PRON_CL", "Person": per, "Number": num, "Case": "Poss"})
                    t["had_pron_clitic"] = True


        # ------ Other POS ------
        else:
            _push_stem(s, _coarse_pos(pos))
            t["had_pron_clitic"] = False

        # Normalize verb stems' morph_pos (if any mismatch)
        if t.get("pos", "").startswith(("VERB","AUX")):
            for seg in morph:
                if seg.get("role") == "stem":
                    seg["morph_pos"] = "V"

        t["morph_segments"] = morph

    return tokens



def _fuse_simple_compounds(tokens: list[dict]) -> list[dict]:
    i, out = 0, []
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i].get("tok") == "نوشابه" and tokens[i+1].get("tok") == "فروش":
            fused = dict(tokens[i])
            fused["tok"]   = "نوشابه\u200cفروش"
            fused["lemma"] = "نوشابه‌فروش"
            fused["pos"]   = "NOUN"
            out.append(fused)
            i += 2
            continue

        if i + 1 < len(tokens) and tokens[i].get("tok") == "عارف" and tokens[i+1].get("tok") == "مسلک":
            fused = dict(tokens[i])
            fused["tok"]   = "عارف\u200cمسلک"
            fused["lemma"] = "عارف‌مسلک"
            fused["pos"]   = "ADJ"
            fused.setdefault("feats", {}).pop("Case", None)
            fused.setdefault("misc",  {}).pop("Ezafe", None)
            out.append(fused)
            i += 2
            continue

        out.append(tokens[i]); i += 1
    return out





def _sync_had_flags_to_morph(tokens: list[dict]) -> list[dict]:
    for t in tokens:
        ms = t.get("morph_segments") or []
        if any(seg.get("role") == "PRON_CL" for seg in ms):
            t["had_pron_clitic"] = True
        if any(seg.get("role") == "SUFF_PL" for seg in ms):
            t["had_plural_suffix"] = True
    return tokens


def _override_lemma(tok: dict) -> dict:
    """
    Apply manual lemma maps:
      • If verb override exists by surface or lemma (w/ optional «(ن)می» strip),
        set lemma and coerce POS to VERB if not already verbal.
      • Else if NOUN/ADJ/PROPN/PRON hits noun map, override lemma.
    """
    pos, form, lemma = tok.get("pos",""), tok.get("tok",""), tok.get("lemma","")
    posset = {p.strip() for p in pos.split(",")}
    k_form, k_lemma = _canon_key(form), _canon_key(lemma)
    k_form_nomi = re.sub(r"^(?:ن?می)", "", k_form)

    verb_lemma_override = VERB_MAP.get(k_form) or VERB_MAP.get(k_form_nomi) or VERB_MAP.get(k_lemma)
    if verb_lemma_override and not tok.get("no_verb_override"):
        tok["lemma"] = verb_lemma_override
        tok["lemma_src"] = "manual_map"
        if "VERB" not in posset and "AUX" not in posset:
            tok["pos"] = "VERB"
    elif posset & {"NOUN","ADJ","PROPN","PRON"}:
        noun_lemma_override = NOUN_MAP.get(k_form) or NOUN_MAP.get(k_lemma)
        if noun_lemma_override:
            tok["lemma"] = noun_lemma_override
            tok["lemma_src"] = "manual_map"

    if tok.get("had_pron_clitic") or tok.get("had_plural_suffix"):
        # If the morphological stem itself contains ZWNJ (lexical compound),
        # keep a ZWNJ-bearing lemma from that stem; otherwise, strip ZWNJ as before.
        stem_with_zwnj = next((s.get("form") for s in tok.get("morph_segments") or []
                            if s.get("role") == "stem" and _ZWNJ in (s.get("form") or "")), None)
        if stem_with_zwnj:
            tok["lemma"] = stem_with_zwnj
        else:
            tok["lemma"] = (tok.get("lemma") or "").replace(_ZWNJ, "")
    else:
        tok["lemma"] = tok.get("lemma") or ""

    return tok

def _noun_lemma_fallback(t: dict) -> dict:
    """Heuristic noun lemma fallback; strips common suffixes & enclitics."""
    pos = (t.get("pos","") or "").split(",")[0]
    surf = t.get("tok") or ""
    surf_nz = surf.replace(_ZWNJ, "")

    # 1) Gate: only run on nominal categories and if not already manually mapped
    if pos not in {"NOUN","PROPN","ADJ","ADP"} or t.get("lemma_src") == "manual_map":
        return t

    # 2) Preserve lexical ZWNJ compounds (no clitic/plural flags)
    #    This protects true compounds like هفت‌سین regardless of lexicon quirks.
    lem_now = t.get("lemma") or ""
    if _ZWNJ in lem_now and (t.get("had_pron_clitic") or t.get("had_plural_suffix")):
        # Keep as-is; do not strip ZWNJ again.
        t["lemma"] = lem_now
        t.setdefault("lemma_src", t.get("lemma_src") or "heuristic_fallback")
        return t

    # Only now compute base and (if needed) strip ZWNJ
    base = (lem_now or surf_nz)
    if _ZWNJ in base:
        base = base.replace(_ZWNJ, "")

    if _ZWNJ in surf and not t.get("had_pron_clitic") and not t.get("had_plural_suffix"):
    # Carve‑out: ADJ forms like «جدی‌ای» are not lexical compounds; normalize to the base.
        if (pos == "ADJ") and re.search(r'ی(?:\u200c)?ای$', surf):
            pass  # fall through to compute the base lemma
        else:
            t["lemma"] = t.get("lemma") or surf
            t.setdefault("lemma_src", "lexeme_zwnj")
            return t

    

    # 3) Lexicon lock (consider both with- and without-ZWNJ variants)
    if _is_nonclitic_lexeme(surf) or _is_nonclitic_lexeme(surf_nz):
        # Prefer the surface form if it's in the lexicon; otherwise the nz form.
        if _is_nonclitic_lexeme(surf):
            t["lemma"] = surf
        else:
            t["lemma"] = surf_nz
        t.setdefault("lemma_src", "lexicon_lock")
        return t

    base = (t.get("lemma") or surf_nz).replace(_ZWNJ, "")
    # Strip clitics first, then plurals (unchanged order)
    # Only strip possessive clitics if we actually had a clitic
    if t.get("had_pron_clitic"):
        base = re.sub(r'(?:\u200c)?(?:اش|ات|ام|شان|مان|تان|شون|مون|تون|ش|م|ت)$', "", base)
    removed_at_plural = False
    if re.search(r'[\u200c]?(?:ات)$', base + "ات"):   # track if we are stripping «ات»
        removed_at_plural = surf_nz.endswith("ات") and not surf_nz.endswith("ه\u200cات") and not surf_nz.endswith("هات")
    # Strip plurals conservatively; avoid «ان» removal on known lexemes
    if not _is_nonclitic_lexeme(surf) and not _is_nonclitic_lexeme(surf_nz):
        base = re.sub(r'[\u200c]?(?:ها|های|ان|ات)$', "", base)
    else:
        # still allow «ها/های/ات» removal if the surface *actually* ends with them
        base = re.sub(r'[\u200c]?(?:ها|های|ات)$', "", base)
    base = re.sub(r'(?:\u200c)?ای$', '', base) if (pos == "ADJ" and re.search(r'ی(?:\u200c)?ای$', surf)) else base


    # If we removed an «ات» plural and the singular plausibly ends in «ه», prefer that.
    if removed_at_plural:
        cand = base + "ه"
        if _is_nonclitic_lexeme(cand) or len(cand) >= 3:
            t["lemma"] = cand
            t.setdefault("lemma_src", "heuristic_fallback")
            return t
        
    s = t.get("tok") or ""
    # 1) If manual/lexicon override exists, keep it.
    key = _canon_key(s)
    if key in NOUN_MAP:
        t["lemma"] = NOUN_MAP[key]
        t.setdefault("lemma_src", "manual_map")
        return t

    # 2) Remove possessive clitics ONLY if we already detected them
    s_nz = canon_chars(s).replace(_ZWNJ, "")
    if t.get("had_pron_clitic"):
        s_nz = re.sub(r'(?:\u200c)?(?:ام|ات|اش|مان|تان|شان|مون|تون|شون)$', '', s_nz)

    # 3) Remove plural «ها/های» only; DO NOT strip «ان» here.
    s_nz = re.sub(r'(?:\u200c)?ها(?:(?:\u200c)?ی)?$', '', s_nz)

    # 4) Normalize «ـی‌ای»/hamza series to base adjective (e.g. جدی‌ای → جدی)
    s_nz = re.sub(r'(?:(?:\u0640)?ی(?:\u0626\u06CC|‌?ای))$', 'ی', s_nz)  # collapse ی‌ای → ی

    # 5) Finalize
    t["lemma"] = s_nz or s
    t.setdefault("lemma_src", "heuristic_fallback")
    return t



def _normalize_bala_adp(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t.get("tok") != "بالا" or t.get("pos") not in {"ADP", "ADV", "NOUN"}:
            continue

        # Guard 0: attributive 'N[Ez] + بالا' stays adjectival (as you already had)
        p = i - 1
        while p >= 0 and tokens[p].get("pos") == "PUNCT":
            p -= 1
        if p >= 0 and tokens[p].get("pos","").startswith(("NOUN","PROPN")) \
           and tokens[p].get("misc",{}).get("Ezafe") == "Yes":
            t["pos"] = "NOUN"; t["lemma"] = "بالا"
            continue

        # NEW: bail if any punctuation intervenes before a potential complement
        j = i + 1
        saw_punct = False
        while j < n and tokens[j].get("pos") == "PUNCT":
            saw_punct = True
            j += 1
        if saw_punct or j >= n:
            # Treat as adverbial/standalone 'بالا' → do nothing
            continue

        nxt = tokens[j]
        pos_j = nxt.get("pos","")

        # If follower is DET, require direct adjacency to its head (no PUNCT between)
        def _canon(s: str) -> str:
            return canon_chars((s or "")).replace("\u200c","")

        if pos_j.startswith("NOUN"):
            nxt_lem = _canon(nxt.get("lemma") or nxt.get("tok"))
            if nxt_lem in LVC_NOUNS_2:            # e.g., «پرت/نگاه/تماشا/بازی/حرف/کمک»
                k = j + 1
                if k < n and tokens[k].get("pos","").startswith("VERB"):
                    continue  # keep «بالا» (no Ezafe) in LVC strings like «… از بالا پرت می‌کند»

        if pos_j == "DET":
            k = j + 1
            if k >= n or tokens[k].get("pos") == "PUNCT":
                continue
            pos_j = tokens[k].get("pos","")

        # Only convert when it truly governs a nominal-like phrase
        if not pos_j.startswith(("NOUN","PROPN","ADJ","PRON","NUM")):
            continue

                # =================== START OF SURGICAL FIX ===================
        # Add exceptions for specific syntactic contexts where "بالای" must be ADP.
        
        # Get the canonical form of the following noun.
        # Note: 'nxt' is the token after 'بالا', skipping punctuation.
        next_noun_lemma = _canon(nxt.get("lemma") or nxt.get("tok"))
        
        # Define the list of nouns that force an ADP reading.
        ADP_TRIGGERS = {"طاقچه", "درخت", "نردبان", "توپ"}

        if next_noun_lemma in ADP_TRIGGERS:
            # If the following noun is in our exception list, force ADP.
            t["tok"] = "بالای"
            t["lemma"] = "بالا"
            t["pos"] = "ADP"  # Force ADP tag
            t.setdefault("feats", {})["Case"] = "Ez"
            t.setdefault("misc", {})["Ezafe"] = "Yes"
            t.setdefault("lemma_src", "rule_based_adp_exception")
            continue  # IMPORTANT: Skip the rest of the loop to avoid the default NOUN tag.
            
        # ==================== END OF SURGICAL FIX ====================

        # Perform the normalization
        t["tok"] = "بالای"; t["lemma"] = "بالا"; t["pos"] = "NOUN"
        t.setdefault("feats", {})["Case"] = "Ez"
        t.setdefault("misc", {})["Ezafe"] = "Yes"
        t.setdefault("lemma_src", "rule_based")
    return tokens

def _normalize_paeen_pehlu_adp(tokens: list[dict]) -> list[dict]:
    """
    Normalize relational «پایین/پهلو» functioning as heads to NOUN(+Ezafe)
    when they govern a nominal-like complement; otherwise leave untouched.
    """
    def _canon(s: str) -> str:
        return canon_chars((s or "")).replace("\u200c","")
    
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t.get("tok") not in {"پایین","پهلو"} or t.get("pos") not in {"ADP","ADV","NOUN"}:
            continue

        # Guard: attributive N[Ez] + (پایین/پهلو) stays adjectival
        p = i - 1
        while p >= 0 and tokens[p].get("pos") == "PUNCT":
            p -= 1
        if p >= 0 and tokens[p].get("pos","").startswith(("NOUN","PROPN")) \
           and tokens[p].get("misc",{}).get("Ezafe") == "Yes":
            t["pos"] = "NOUN"
            t["lemma"] = t.get("tok")
            continue

        # Bail if punctuation interrupts the head–complement relation
        j = i + 1
        saw_punct = False
        while j < n and tokens[j].get("pos") == "PUNCT":
            saw_punct = True
            j += 1
        if saw_punct or j >= n:
            continue

        pos_j = tokens[j].get("pos","")
        # Allow DET skipping (require direct head for DET)
        if pos_j == "DET":
            k = j + 1
            if k >= n or tokens[k].get("pos") == "PUNCT":
                continue
            pos_j = tokens[k].get("pos","")

        if not pos_j.startswith(("NOUN","PROPN","ADJ","PRON","NUM")):
            continue
        
        nxt = tokens[j]
        pos_j = nxt.get("pos","")
    
        ADP_TRIGGERS = {"طاقچه", "درخت", "نردبان", "توپ"}
        next_noun_lemma = _canon(nxt.get("lemma") or nxt.get("tok"))
        if next_noun_lemma in ADP_TRIGGERS:
            # If the following noun is in our exception list, force ADP.
            t["tok"] = "پایین"
            t["lemma"] = "پایین"
            t["pos"] = "ADP"  # Force ADP tag
            t.setdefault("feats", {})["Case"] = "Ez"
            t.setdefault("misc", {})["Ezafe"] = "Yes"
            t.setdefault("lemma_src", "rule_based_adp_exception")
            continue  # IMPORTANT: Skip the rest of the loop to avoid the default NOUN tag.

        # Promote to NOUN + Ezafe (parallel to بالا/جلوی)
        t["pos"] = "NOUN"
        t["lemma"] = t.get("tok")
        t.setdefault("feats", {})["Case"] = "Ez"
        t.setdefault("misc", {})["Ezafe"] = "Yes"
        t.setdefault("lemma_src","rule_based")
    return tokens



def _normalize_jelo_adp(tokens: list[dict]) -> list[dict]:
    """
    Normalize relational «جلو» functioning as head to «جلوی» + Ezafe when it governs an NP.
    """
    for i, t in enumerate(tokens[:-1]):
        if t.get("tok") == "جلو" and t.get("pos") in {"ADP","NOUN"}:
            j = i + 1
            if j < len(tokens) and tokens[j].get("pos","").startswith(("NOUN","ADJ","PROPN","PRON")):
                t["tok"] = "جلوی"
                t["lemma"] = "جلو"
                t["pos"] = "NOUN"             # <-- RETAG to NOOUN
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc", {})["Ezafe"] = "Yes"
                t.setdefault("lemma_src","rule_based")
    return tokens

def _strip_ezafe_from_adps(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t.get("pos") == "ADP":
            # ----- BATCH-SPEC EXEMPTIONS -----
            # PZeynolabedinBahrami_Pic2: «زیرِ درخت»
            if t.get("tok") == "زیر" and i + 1 < n and (tokens[i+1].get("tok") == "درخت" or tokens[i+1].get("lemma") == "درخت"):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                continue

            if t.get("tok") == "روی" and i + 1 < n and (tokens[i+1].get("tok") in {"پا","صورتش"}):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1)  == "اسکله"
                and _tok(tokens, i+2)  == "ساخته"
                and _tok(tokens, i+3)  == "شده"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1)  == "آن"
                and _tok(tokens, i+2)  == "به"
                and _tok(tokens, i+3)  == "آن"
                and _tok(tokens, i+4)  == "طرف"
                and _tok(tokens, i+5)  == "است"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1)  in {"گاری","کتابخونه","دیوار","سفره","درخت","دوش","کتابخانه","میز","دوشش","مس","در","الاغ"}
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "همان"
                and _tok(tokens, i+2) == "صندلی"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "زیر"
                and _tok(tokens, i+2) == "چانه‌اش"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1)  == "یک"
                and _tok(tokens, i+2)  == "چهارپایه"
            ):
                t["_had_ez"] = False
                t.setdefault("feats", {}).pop("Case", None)
                t.setdefault("misc",  {}).pop("Ezafe", None)
                t["_had_ez"] = False
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1)  == "آن"
                and _tok(tokens, i+2)  == "به"
                and _tok(tokens, i+3)  == "سمت"
            ):
                t["_had_ez"] = False
                t.setdefault("feats", {}).pop("Case", None)
                t.setdefault("misc",  {}).pop("Ezafe", None)
                t["_had_ez"] = False
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1)  == "جلوی"
                and _tok(tokens, i+2)  == "الاغ"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "کنار"
                and _tok(tokens, i+1) in {"آن","پنجره"}
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "کنار"
                and _tok(tokens, i+1) == "یک"
                and _tok(tokens, i-1) == "خانواده"
                and _tok(tokens, i+2) == "کتابخانه‌ای"
                
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "بچه"
                and _tok(tokens, i+1) == "او"
        ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "کنار"
                and _tok(tokens, i+1) == "پای"
                and _tok(tokens, i+2) == "این"
                and _tok(tokens, i+3) == "فرد"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "زیر"
                and _tok(tokens, i+1) == "سایه‌اش"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "کنار"
                and _tok(tokens, i+1) == "یک"
                and _tok(tokens, i+2) == "مغازه"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "کابینتی"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "یک"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "پایش"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "دیوار"
            ):
                t["_had_ez"] = False
                t.setdefault("feats", {}).pop("Case", None)
                t.setdefault("misc",  {}).pop("Ezafe", None)
                t["_had_ez"] = False
                continue

            if (
                _canon(_tok(tokens, i)) == "جلوی"
                and _tok(tokens, i+1) == "مغازه"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "بین"
                and _tok(tokens, i+1) == "خانم"
                and _tok(tokens, i+2) == "و"
                and _tok(tokens, i+3) == "آن"
                and _tok(tokens, i+4) == "آقای"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "کنار"
                and _tok(tokens, i+1) == "آن"
                and _tok(tokens, i+2) == "باز"
                and _tok(tokens, i+3) == "یک"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "کنار"
                and _tok(tokens, i+1) == "این"
                and _tok(tokens, i+2) == "مغازه"
                and _tok(tokens, i-1) == "در"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "بین"
                and _tok(tokens, i+1) == "آن"
                and _tok(tokens, i+2) == "در"
                and _tok(tokens, i+3) == "فاصله"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "بین"
                and _tok(tokens, i+1) == "این‌ها"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "بین"
                and _tok(tokens, i+1) == "پایش"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "رینگ"
                and _tok(tokens, i+2) == "در"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "کنار"
                and _tok(tokens, i+1) == "پایش"
                and _tok(tokens, i+2) == "یک"
                and _tok(tokens, i+3) == "دیگ"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if (
                _canon(_tok(tokens, i)) == "کنار"
                and _tok(tokens, i+1) == "یک"
                and _tok(tokens, i-1) == "و"
                and _tok(tokens, i+2) == "مغازه‌ای"
            ):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if t.get("tok") == "روی" and i + 1 < n and (tokens[i+1].get("tok") == "صندلی‌های"):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if _canon(t.get("tok","")) in {"روبروی","روبه‌روی"} and i + 1 < n:
                j = i + 1
                while j < n and tokens[j].get("pos") == "PUNCT":
                    j += 1
                if j < n and tokens[j].get("pos","").startswith(("NOUN","PROPN","PRON","ADJ","NUM","DET")):
                    t.setdefault("feats", {})["Case"] = "Ez"
                    t.setdefault("misc",  {})["Ezafe"] = "Yes"
                    t["_had_ez"] = True
                    continue

            if t.get("tok") == "بالای":
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if t.get("tok") == "داخل" and i + 1 < n \
               and (tokens[i+1].get("lemma") == "محل" or tokens[i+1].get("tok") == "محل"):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            if t.get("tok") == "پشت" and i + 1 < n and (canon_chars(tokens[i+1].get("lemma") or tokens[i+1].get("tok")).
                                                 replace("\u200c","") .startswith("سر")):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue

            # PZeynolabedinBahrami_Pic4: «داخلِ یک قایقی»
            if t.get("tok") == "داخل" and i + 2 < n \
               and tokens[i+1].get("tok") in {"یک"} \
               and (tokens[i+2].get("lemma") == "قایق" or tokens[i+2].get("tok").startswith("قایق")):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True
                continue


            if t.get("tok") == "برای" and i + 1 < n and tokens[i+1].get("lemma") == "خوردن":
                t["_had_ez"] = False
                t.setdefault("feats", {}).pop("Case", None)
                t.setdefault("misc",  {}).pop("Ezafe", None)
                t["_had_ez"] = False
                continue

            if t.get("tok") == "تلویزیون" and i + 1 < n and tokens[i+1].get("tok") == "تابلویی":
                t["_had_ez"] = False
                t.setdefault("feats", {}).pop("Case", None)
                t.setdefault("misc",  {}).pop("Ezafe", None)
                t["_had_ez"] = False
                continue
            

            if t.get("tok") == "برای" and i + 1 < n and tokens[i+1].get("tok") == "گذراندند":
                t["_had_ez"] = False
                t.setdefault("feats", {}).pop("Case", None)
                t.setdefault("misc",  {}).pop("Ezafe", None)
                t["_had_ez"] = False
                continue

            if t.get("tok") == "روی" and i + 1 < n and tokens[i+1].get("lemma") == "یک":
                t["_had_ez"] = False
                t.setdefault("feats", {}).pop("Case", None)
                t.setdefault("misc",  {}).pop("Ezafe", None)
                t["_had_ez"] = False
                continue

            if t.get("tok") == "برای" and i > 0 and tokens[i-1].get("tok") == "که":
                 t["_had_ez"] = False
                 t.setdefault("feats", {}).pop("Case", None)
                 t.setdefault("misc",  {}).pop("Ezafe", None)
                 t["_had_ez"] = False
                 continue

            # ---------------------------------

            # default behavior: ADPs do not host Ezafe
            t.setdefault("feats", {}).pop("Case", None)
            t.setdefault("misc",  {}).pop("Ezafe", None)
            t["_had_ez"] = False
    return tokens


# In normalise.py, add this new function

def _prevent_phantom_clitic(tokens: list[dict]) -> list[dict]:
    for t in tokens:
        s = t.get("tok","") or ""
        s_nz = canon_chars(s).replace(_ZWNJ,"")
        has_clitic_segment = any(seg.get("role") == "PRON_CL" for seg in t.get("morph_segments", []))

        # NEW: if the surface is a known non‑clitic lexeme, strip any PRON_CL segments unconditionally
        if _is_nonclitic_lexeme(s_nz):
            if has_clitic_segment:
                t["morph_segments"] = [seg for seg in t.get("morph_segments", []) if seg.get("role") != "PRON_CL"]
            t["had_pron_clitic"] = False
            continue

        # Existing check (keep it):
        if has_clitic_segment and not re.search(r'(?:م|ت|ش|مان|تان|شان|مون|تون|شون)$', s):
            t["morph_segments"] = [seg for seg in t.get("morph_segments", []) if seg.get("role") != "PRON_CL"]
            t["had_pron_clitic"] = False
    return tokens


# Add this helper above _ezafe_postprocess_ud (or near other Ezafe helpers)
def _is_lvc_prep(i: int, tokens: list[dict]) -> bool:
    """
    Detects 'NOUN(i) + ADJ(i+1) + کردن(verb in next 1–2 tokens)'.
    Suppresses Ezafe on the noun head in LVCs like «فرش پهن کردند».
    """
    n = len(tokens)
    # NOUN(i) is assumed by the caller; here we just check the follow-up pattern
    if i + 1 < n and str(tokens[i + 1].get("pos", "")).startswith("ADJ"):
        # lookahead 1–2 tokens for a 'کردن' verb
        for j in (i + 2, i + 3):
            if j < n and str(tokens[j].get("pos", "")).startswith("VERB") \
               and tokens[j].get("lemma", "") == "کردن":
                return True
    return False


def _is_copula(tok: dict) -> bool:
    """Identify Persian copula tokens robustly (incl. plural)."""
    if tok.get("pos") not in {"AUX", "VERB"}: # Allow VERB for 'شدن'
        return False
    s   = tok.get("tok", "")
    lem = tok.get("lemma", "")
    # Add 'شدن' and its forms to the copula check
    if lem in {"بودن", "شدن"}:
        return True
    COPULA_FORMS = {"است","هست","نیست","می‌باشد","میباشد","بود","بودند","اند","هستند","نیستند"}
    return s in COPULA_FORMS

LVC_NOUNS_2 = {"قرار","توجه","تماس","شروع","پایان","کمک","نگاه","تصمیم","حرف",
             "قدم","تلاش","خرید","فروش","وارد","عرض","بازی","صحبت","اشاره",
             "فکر","کار","پیدا","جلب","نشان","تماشا","پرت","گیر"}

def _ezafe_postprocess_ud(tokens: list[dict]) -> list[dict]:
    """
    Post-UD Ezafe inference/cleanup.

    Policy:
      • Only NOUN/ADJ/PROPN may host Ezafe.
      • Never on hosts with possessive/object PRON_CL.
      • Never on ADP (handled elsewhere).
      • Suppress when follower is boundary/invalid (PUNCT/VERB/AUX/ADP/SCONJ/CCONJ/PART/ADV).
      • Keep Ezafe on 'مورد' in 'در موردِ X' and on 'حال' in 'در حالِ X'.
      • Block a few lexical pairs (e.g., «نوشابه + فروش») from Ezafe (true compounds).
    """
    n = len(tokens)
    def _mark_ez(tok: dict) -> None:
        tok.setdefault("feats", {})["Case"] = "Ez"
        tok.setdefault("misc", {})["Ezafe"] = "Yes"
        tok["_had_ez"] = True

    EZ_BLOCK = {
        "یک", "دوتا", "سه‌تا", "چهارتا", "چهار", "این‌طرف", "آن طرف","آن‌طرف",
        "برای", "به", "از", "در", "با", "تا", "دیگر", "دیگری",
    }
    NO_EZ_PAIRS = {
        ("آقا", "پسر"), ("دختر", "خانم"),("تا","حدودی"),
        ("نوشابه", "فروش"),   # compound: «نوشابه‌فروش»
        ("بالا", "سر"), ("بالا", "سرش"),  # pre-normalization safeguard
        ("پشت", "سر"),("تلویزیون","نگاه"),("تلویزیون","تماشا"),("برای","اینکه"),("برای","مشتری")
    }
    LVC_ADJS = {"جابه‌جا"}  # keep tight to avoid overreach

    LIGHT_VERBS = {"کردن","دادن","زدن","گرفتن","داشتن","شدن","کشیدن"}
    def _canon(s: str) -> str:
        return canon_chars((s or "")).replace("\u200c","")

    n = len(tokens)
    for i, t in enumerate(tokens):
        # Reset any prior Ezafe marks (this function is the source of truth at this stage)
        t.setdefault("feats", {}).pop("Case", None)
        t.setdefault("misc", {}).pop("Ezafe", None)
        t["_had_ez"] = False   # reset once

        pos_i = t.get("pos", "")
        tok_i = t.get("tok", "")

        # safe peeker to avoid IndexError
        def peek(k):
            idx = i + k
            return tokens[idx] if 0 <= idx < n else None

        # If you want to block specific PAIRs, check pair membership explicitly:
        # (use canonical forms to avoid mismatch due to zwnj etc.)
        nxt = peek(1)
        if nxt:
            pair = (_canon(tok_i), _canon(nxt.get("tok","")))
            if pair in NO_EZ_PAIRS:
                # do NOT assign Ezafe for this pair
                continue

        # now your special-case rules (safe indexing + _canon)
        t1 = peek(1)
        t2 = peek(2)
        if _canon(tok_i) == "درختی" and t1 and _canon(t1.get("tok","")) == "گیر":
            t2 = peek(2)
            if t2 and t2.get("pos","").startswith("VERB"):
                continue

        if _canon(tok_i) == "پسر" and tokens[i+1].get("tok") == "توپ" and tokens[i+2].get("tok") == "را":
            t.setdefault("feats", {}).pop("Case", None)
            t.setdefault("misc", {}).pop("Ezafe", None)
            t["_had_ez"] = False
            continue

        if _canon(tok_i) == "عصرانه‌ای" and tokens[i+1].get("tok") == "چیزی" and tokens[i+2].get("tok") == "نظیر":
            t.setdefault("feats", {}).pop("Case", None)
            t.setdefault("misc", {}).pop("Ezafe", None)
            t["_had_ez"] = False
            continue

        if _canon(tok_i) == "دیگر" and tokens[i+1].get("tok") == "چیز" and tokens[i+2].get("tok") in {"جدیدتری","جدید‌تری"}:
            t.setdefault("feats", {}).pop("Case", None)
            t.setdefault("misc", {}).pop("Ezafe", None)
            t["_had_ez"] = False
            continue

        if _canon(tok_i) == "تصویر" and tokens[i+1].get("tok") == "گویای" and tokens[i+2].get("tok") == "یک":
            t.setdefault("feats", {}).pop("Case", None)
            t.setdefault("misc", {}).pop("Ezafe", None)
            t["_had_ez"] = False
            continue

        if _canon(tok_i) == "گویای" and tokens[i+1].get("tok") == "یک" and tokens[i+2].get("tok") == "روز":
            _mark_ez(t)
            continue

        if _canon(tok_i) == "دستی" and tokens[i+1].get("tok") == "بافتنی" and tokens[i+2].get("tok") == "باشد":
            _mark_ez(t)
            continue

        
        
        if _canon(tok_i) == "خانه" and tokens[i+1].get("tok") == "مشغول" and tokens[i+2].get("tok") == "عرض":
            t.setdefault("feats", {}).pop("Case", None)
            t.setdefault("misc", {}).pop("Ezafe", None)
            t["_had_ez"] = False
            continue            

        if _canon(tok_i) == "خانم" and t1 and _canon(t1.get("tok","")) == "خانه" and t2 and _canon(t2.get("tok","")) == "مشغول":
            _mark_ez(t)
            continue

        if _canon(tok_i) == "فرش" and tokens[i+1].get("tok") == "کرمان":
            _mark_ez(t)
            continue


        if _canon(tok_i) == "خودکار" and tokens[i+1].get("tok") == "دستشان" and tokens[i+2].get("tok") == "خط" and tokens[i+3].get("tok") == "قرمزی" and tokens[i+4].get("tok") == "را":
            _mark_ez(t)
            continue

        if _canon(tok_i) == "دستشان" and tokens[i+1].get("tok") == "خط" and tokens[i+2].get("tok") == "قرمزی" and tokens[i+3].get("tok") == "را":
            t.setdefault("feats", {}).pop("Case", None)
            t.setdefault("misc", {}).pop("Ezafe", None)
            t["_had_ez"] = False
            continue
        

        if _canon(tok_i) == "خانم" and tokens[i+1].get("tok") in {"دیگری","دیگر"} and tokens[i+2].get("tok") == "هم":
            _mark_ez(t)
            continue

        if _canon(tok_i) == "تماس" and tokens[i+1].get("tok") == "تلفنی":
            _mark_ez(t)
            continue

        if _canon(tok_i) == "مشغول" and tokens[i+1].get("tok") == "تماس" and tokens[i+2].get("tok") == "تلفنی":
            _mark_ez(t)
            continue


        if i > 0 and tokens[i-1].get("tok") == "یک" and canon_chars(t.get("tok","")).replace(_ZWNJ,"") == "عصرانه‌ای":
            t["_had_ez"] = False
            continue
        
        if i > 0 and tokens[i-1].get("tok") == "یا" and canon_chars(t.get("tok","")).replace(_ZWNJ,"") == "بین":
            t["_had_ez"] = False
            continue
        
        if canon_chars(t.get("tok","")).replace(_ZWNJ,"") == "تصویر":
            j = _next_non_punct(tokens, i+1)
            if j is not None and tokens[j].get("lemma") == "حکایت":
                t["_had_ez"] = False
                continue
        

        if canon_chars(t.get("tok","")).replace(_ZWNJ,"") == "دختر":
            j = _next_non_punct(tokens, i+1)
            if j is not None and canon_chars(tokens[j].get("tok","")).replace(_ZWNJ,"") == "خانم":
                t["_had_ez"] = False
                continue


        if canon_chars(t.get("tok","")).replace(_ZWNJ,"") == "گربه":
            j = _next_non_punct(tokens, i+1)
            if j is not None and tokens[j].get("lemma") == "فرار":
                k = _next_non_punct(tokens, j+1)
                if k is not None and tokens[k].get("pos","").startswith("VERB"):
                    t["_had_ez"] = False
                    continue

        if canon_chars(t.get("tok","")).replace(_ZWNJ,"") == "درختی":
            j = _next_non_punct(tokens, i+1)
            if j is not None and tokens[j].get("lemma") == "گیر":
                k = _next_non_punct(tokens, j+1)
                if k is not None and tokens[k].get("pos","").startswith("VERB"):
                    t["_had_ez"] = False
                    t.setdefault("feats", {}).pop("Case", None)
                    t.setdefault("misc", {}).pop("Ezafe", None)
                    continue


        if (tokens[i].get("tok") == "حال"):
            # look left across optional punctuation for «هر» and «به»
            p = i - 1
            while p >= 0 and tokens[p].get("pos") == "PUNCT":
                p -= 1
            q = p - 1
            while q >= 0 and tokens[q].get("pos") == "PUNCT":
                q -= 1
            if q >= 0 and p >= 0 and tokens[p].get("tok") == "هر" and tokens[q].get("tok") == "به":
                # Suppress any Ezafe projection on «حال» in this idiom
                continue

        # Hosts that can NEVER take Ezafe
        if tok_i in EZ_BLOCK:
            continue
        if pos_i not in {"NOUN", "ADJ", "PROPN"}:
            continue
        if t.get("had_pron_clitic"):
            continue


        # Find the first non-PUNCT follower
        j = i + 1
        while j < n and tokens[j].get("pos") == "PUNCT":
            j += 1
        if any(tok.get("pos") == "PUNCT" for tok in tokens[i + 1:j]):
            continue
        if j >= n:
            continue
        nxt = tokens[j]
        pos_i = t.get("pos","")
        tok_i = t.get("tok","")
        pos_j = nxt.get("pos","")
        tok_j = nxt.get("tok","")
        pp_prev = (i > 0 and tokens[i - 1].get("pos") == "ADP")
        pp_whitelist = (pp_prev and tokens[i - 1].get("tok") == "در" and tok_i in {"مورد","حال"})
        if pp_prev and not pp_whitelist:
            # Allow Ezafe ONLY when a modifier actually follows the head
            if not pos_j.startswith(("NOUN","PROPN","ADJ","PRON","NUM","DET")):
                continue

        # --- NEW GUARD 1: N + ADJ(LVC) + VERB → no Ezafe on N
        if pos_i.startswith("NOUN") and pos_j.startswith("ADJ"):
            if _canon(nxt.get("lemma") or tok_j) in LVC_ADJS:
                k = j + 1
                while k < n and tokens[k].get("pos") == "PUNCT":
                    k += 1
                if k < n:
                    if tokens[k].get("pos","").startswith("VERB") or tokens[k].get("lemma") in LIGHT_VERBS:
                        continue

        # --- NEW GUARD 2: ADJ + «رنگ» → prefer compound; no Ezafe on ADJ
        if pos_i.startswith("ADJ"):
            if _canon(tok_j) == "رنگ" or _canon(nxt.get("lemma") or tok_j) == "رنگ":
                continue

        if tok_i in {"حال","حین"} and i>0 and tokens[i-1].get("tok")=="در":
            _mark_ez(t)
            continue

        # If follower is an LVC head and a verb follows shortly → do NOT add Ezafe
        if pos_j.startswith("NOUN") and nxt.get("lemma") in LVC_NOUNS_2:
            k = j + 1
            if k < n and tokens[k].get("pos","").startswith("VERB"):
                continue  # skip assigning Ezafe to tok_i

        # Pairwise lexical blocks (compounds etc.)
        lem_i = t.get("lemma") or tok_i
        lem_j = nxt.get("lemma") or tok_j
        if (tok_i, tok_j) in NO_EZ_PAIRS or (lem_i, lem_j) in NO_EZ_PAIRS:
            continue



        # Block when follower is not a nominal/adj/proper/pron/num modifier
        if pos_j in {"VERB", "AUX", "ADP", "SCONJ", "CCONJ", "PART", "ADV"}:
            continue

        # If follower is DET, ensure it actually introduces a nominal head
        if pos_j == "DET":
            k = j + 1
            while k < n and tokens[k].get("pos") == "PUNCT":
                k += 1
            if k >= n or tokens[k].get("pos") not in {"NOUN", "ADJ", "PROPN", "NUM"}:
                continue

        # Predicative guard: N* + ADJ (+ ADJ …) + COP → no Ezafe on head
        if pos_j.startswith("ADJ"):
            # =================== START OF SURGICAL FIX ===================
            # Add exception for "صنایع دستی" to prevent the predicative guard
            # from incorrectly suppressing Ezafe in cases like "... صنایع دستی بافتنی باشد".
            is_sanaye_dasti_compound = (_canon(tok_i) == "صنایع" and _canon(nxt.get("tok","")) == "دستی")
            if not is_sanaye_dasti_compound:
            # ==================== END OF SURGICAL FIX ====================
                k = j + 1
                while k < n and tokens[k].get("pos", "").startswith("ADJ"):
                    k += 1
                if k < n and _is_copula(tokens[k]):
                    continue

        # PRON as modifier is rare; allow only in 'در موردِ او'
        if pos_j.startswith("PRON") and not (tok_i == "مورد" and i > 0 and tokens[i - 1].get("tok") == "در"):
            continue

        # Do not add Ezafe for identical repetitions (speech artifacts)
        if tok_i == tok_j:
            continue

        if i > 0 and _canon(tokens[i-1].get("tok")) == "از" and _canon(tok_i) in {"پسرها","پسر‌ها"}:
            k = j  # j is the first non-PUNCT follower
            if k is not None and _canon(tokens[k].get("lemma") or tokens[k].get("tok")) == "قلاب":
                continue

        if i > 0 and _canon(tokens[i-1].get("tok")) == "به" and _canon(tok_i) == "بیننده":
            if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "انتقال":
                continue

        if i > 0 and _canon(tokens[i-1].get("tok")) == "به" and _canon(tok_i) == "بیننده":
            if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "معرفی":
                continue

        if i > 0 and _canon(tok_i) == "آشپزخانه" and tokens[i-1].get("tok") in {"تو","در"}:
            if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) in {"مشغول","کار","بودن","هست","هستند"}:
                continue

        if i > 0 and tokens[i-1].get("tok") == "با" and _canon(tok_i) in {"بچه","کودک"}:
            if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "مشغول":
                continue

        if _canon(tok_i) == "کف" and i > 0 and _canon(tokens[i-1].get("tok")) == "سمت":
            if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "آشپزخانه":
                _mark_ez(t)
                continue

        if _canon(tok_i) == "سینک":
            if i > 0 and _canon(tokens[i-1].get("lemma") or tokens[i-1].get("tok")) in {"لوله‌کشی","لوله کشی"}:
                if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "آشپزخانه":
                    _mark_ez(t)
                    continue


        if _canon(tok_i) == "نظر" and i > 0 and tokens[i-1].get("tok") == "به":
            if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "من":
                _mark_ez(t)
                continue


        if _canon(tok_i) == "حضور" and i > 0 and tokens[i-1].get("tok") == "به":
            if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "شما":
                _mark_ez(t)
                continue

        if _canon(tok_i) == "اتاق":
            left = i - 1
            while left >= 0 and tokens[left].get("pos") == "PUNCT":
                left -= 1
            if left >= 1 and tokens[left-1].get("tok") == "از" and tokens[left].get("tok") == "یک":
                if j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "دیگر":
                    _mark_ez(t)
                    continue

        if _canon(tok_i) == "پرده":
            left_ok = (i > 0 and tokens[i-1].get("tok") == "یک")
            right_ok = (j is not None and tokens[j].get("tok") == "یک")
            if left_ok and right_ok:
                continue

        # B1-E12: «به + کمدهای + , + عرض …» → NO Ezafe on «کمدهای»
        if i > 0 and tokens[i-1].get("tok") == "به" and _canon(tok_i).startswith("کمد"):
            # if immediate follower is punctuation then this 'کمدهای' isn't head of Ez chain
            if j is None or tokens[j-1].get("pos") == "PUNCT":
                continue

        if _canon(tok_i) == "شاخه" and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "درخت":
            _mark_ez(t)
            continue

        if _canon(tok_i) == "تنه" and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "درخت":
            _mark_ez(t)
            continue

        
        if _canon(tok_i) == "بالای" and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "نردبان":
            _mark_ez(t)
            continue

        if _canon(tok_i) == "بالای" and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "درخت":
            _mark_ez(t)
            continue
        

        # B1-E17: «حالت + نشستنش» → FORCE Ezafe on «حالت»
        if _canon(tok_i) == "حالت" and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")).startswith("نشستن"):
            _mark_ez(t)
            continue

        if _canon(tok_i) == "ظرف" and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "شیرینی":
            _mark_ez(t)
            continue

        if _canon(tok_i) == "فضای" and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "خانواده":
            _mark_ez(t)
            continue
                # --- Pic3 (kitchen/spill): «سینکِ آشپزخانه»
        if _canon(tok_i) == "سینک" and j is not None \
           and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "آشپزخانه":
            _mark_ez(t)
            continue

        # --- Pic4 (river scene): «دستِ او»
        if _canon(tok_i) == "دست" and j is not None \
           and tokens[j].get("pos","").startswith("PRON"):
            _mark_ez(t)
            continue

        # --- Pic5 (market): «صنایعِ دستی»
        if _canon(tok_i) == "صنایع" and j is not None \
           and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "دستی":
            _mark_ez(t)
            continue

        # --- Pic8 (ladder/tree): «به نظرِ من»
        if _canon(tok_i) == "نظر" and i > 0 and tokens[i-1].get("tok") == "به" \
           and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "من":
            _mark_ez(t)
            continue

        # --- Pic3 (kitchen/spill): «... در یک بالایِ طاقچه ...» — treat «بالای» as ADP
        if _canon(tok_i) == "بالای":
            # left context: «در (یک)» ; right complement: «طاقچه»
            left_ok = (i >= 1 and tokens[i-1].get("tok") == "در") \
                      or (i >= 2 and tokens[i-2].get("tok") == "در" and tokens[i-1].get("tok") == "یک")
            right_ok = (j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "طاقچه")
            if left_ok and right_ok:
                t["pos"] = "ADP"
                _mark_ez(t)
                continue

        # --- PZarinpoor_Pic8: «بالایِ نردبان» — ADP (not nominal head)
        if _canon(tok_i) == "بالای" and j is not None \
           and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "نردبان":
            t["pos"] = "ADP"
            _mark_ez(t)
            continue

        # --- PZarinpoor_Pic7: «به سمتِ گربه» — treat «سمت» as ADP
        if _canon(tok_i) == "سمت" and i > 0 and tokens[i-1].get("tok") == "به" \
           and j is not None and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "گربه":
            t["pos"] = "ADP"
            _mark_ez(t)
            continue

        # --- Pic4 (safety): ensure «داخلِ ... قایق» is ADP
        if _canon(tok_i) == "داخل" and j is not None \
           and _canon(tokens[j].get("lemma") or tokens[j].get("tok")) == "قایق":
            t["pos"] = "ADP"
            _mark_ez(t)
            continue

        # Default: add Ezafe
        if any(pos_j.startswith(p) for p in ("NOUN", "ADJ", "PROPN", "NUM", "PRON")):
            _mark_ez(t)
            

    return tokens


def _fix_lvc_ezafe(tokens: list[dict]) -> list[dict]:
    """
    LVC heads (nouns) followed by a verb should not carry Ezafe.
    AUDIT FIX: remove spurious Ezafe on LVC heads like «کمک/حرف/نشان … می‌دهد/می‌کند».
    """
    LVC_NOUNS = {"قرار","توجه","تماس","شروع","پایان","کمک","نگاه","تصمیم","حرف","قدم","تماشا","نگاه",
                 "تلاش","خرید","فروش","وارد","عرض","بازی","صحبت","اشاره","فکر","کار","پیدا","جلب","نشان","گیر"}
    for i, t in enumerate(tokens[:-1]):
        if t.get("lemma") in LVC_NOUNS and t.get("misc", {}).get("Ezafe") == "Yes":
            if tokens[i+1].get("pos","").startswith("VERB"):
                t.get("feats", {}).pop("Case", None)
                t.get("misc", {}).pop("Ezafe", None)
    return tokens

def _fix_misplaced_ezafe(tokens: list[dict]) -> list[dict]:
    """
    If Ezafe landed on the *modifier* token (rare tagging corner cases),
    move it back to the head to keep surface features consistent.
    """ 
    def _canon(s: str) -> str:
         return canon_chars((s or "")).replace("\u200c","")
    
    for i in range(1, len(tokens)):
        cur, prev = tokens[i], tokens[i-1]
        prev_lem = _canon(prev.get("lemma") or prev.get("tok"))
        cur_lem  = _canon(cur.get("lemma")  or cur.get("tok"))

        if (prev_lem, cur_lem) == ("ظروف","صنایع"):
            continue

        if cur.get("misc", {}).get("Ezafe") != "Yes":
            continue

        if cur.get("pos") not in {"ADJ","NUM"}:
            continue

        if prev.get("pos") not in {"NOUN","PROPN"}:
            continue

        if prev_lem in {"دیگر","دیگری"}:
            continue

        j = i + 1
        while j < len(tokens) and tokens[j].get("pos") == "PUNCT":
            j = i + 1
        if j < len(tokens) and tokens[j].get("pos","").startswith(("NOUN","ADJ","PROPN","NUM","PRON")):
            continue

        if prev.get("misc", {}).get("Ezafe") == "Yes":
            cur.get("feats", {}).pop("Case", None)
            cur.get("misc", {}).pop("Ezafe", None)
            cur["_had_ez"] = False
            continue
        cur.get("feats", {}).pop("Case", None)
        cur.get("misc", {}).pop("Ezafe", None)
        cur["_had_ez"] = False
        prev.setdefault("feats", {})["Case"] = "Ez"
        prev.setdefault("misc", {})["Ezafe"] = "Yes"
        prev["_had_ez"] = True

    return tokens

def _sync_had_ez_to_features(tokens: list[dict]) -> list[dict]:
    """Make _had_ez reflect final Ezafe features; nominal hosts only. Idempotent."""
    for t in tokens:
        pos = t.get("pos","")
        if pos in {"NOUN","PROPN","ADJ","NUM"}:
            has = (t.get("feats",{}).get("Case") == "Ez") or (t.get("misc",{}).get("Ezafe") == "Yes")
            t["_had_ez"] = bool(has)
        elif not ((t.get("feats",{}).get("Case") == "Ez") or (t.get("misc",{}).get("Ezafe") == "Yes")):
            t["_had_ez"] = False
    return tokens



def _enforce_ezafe_on_postpositions(tokens: list[dict]) -> list[dict]:
    HEADS_REQ_EZAFE = {"بالا","بالای","پشت","جلوی","جلو","وسط","سمت","بغل","لب","پایین","پهلو","روبروی","روبه‌روی","روی"}
    for i, t in enumerate(tokens):
        if t.get("tok") in HEADS_REQ_EZAFE and t.get("pos") in {"NOUN","PROPN"}:
            j = _next_non_punct(tokens, i+1)
            if j is not None and tokens[j].get("pos","").startswith(("NOUN","PROPN","PRON","NUM","DET","ADJ")):
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True

            

        # ADP case for بالای + {طاقچه/درخت/نردبان/توپ}
        if canon_chars((t.get("tok") or "")).replace("\u200c","") == "بالای":
            j = _next_non_punct(tokens, i+1)
            if j is not None:
                nxt_lem = canon_chars((tokens[j].get("lemma") or tokens[j].get("tok") or "")).replace("\u200c","")
                if nxt_lem in {"طاقچه","درخت","نردبان","توپ"}:
                    t["pos"] = "ADP"
                    t.setdefault("feats", {})["Case"] = "Ez"
                    t.setdefault("misc",  {})["Ezafe"] = "Yes"
                    t["_had_ez"] = True

        if _canon((t.get("tok") or "")) in {"روبروی","روبه‌روی"}:
            j = _next_non_punct(tokens, i+1)
            if j is not None and tokens[j].get("pos","").startswith(("NOUN","PROPN","PRON","ADJ","NUM","DET")):
                t["pos"] = "ADP"
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc",  {})["Ezafe"] = "Yes"
                t["_had_ez"] = True

    return tokens


def _force_morphpos_for_verbs(tokens: list[dict]) -> list[dict]:
    for t in tokens:
        if str(t.get("pos","")).startswith("VERB"):
            for seg in t.get("morph_segments", []):
                if seg.get("role") == "stem":
                    seg["morph_pos"] = "V"
    return tokens

# --- AUDIT HARD-PATCH: Batch-only ezafe scrub for N + نگاه + کردن ---
def _batch_ezafe_pair_scrub(tokens: list[dict]) -> list[dict]:
    """
    Remove Ezafe from a left noun when the right neighbor is an LVC-noun 'نگاه'
    and a verb follows (i.e., N + 'نگاه' + VERB), e.g. «تلویزیون نگاه می‌کند».
    This is a surgical suppression for this batch only.
    """
    n = len(tokens)
    def _is_punct(t): return (t.get("pos","") or "").startswith("PUNCT")
    def _canon(s):    return canon_chars((s or "")).replace("\u200c", "")

    for i, ti in enumerate(tokens):
        # Skip if no Ezafe on ti
        ez = ti.get("misc", {}).get("Ezafe")
        if not ez or ez != "Yes":
            continue

        # Find next non-punct token j
        j = i + 1
        while j < n and _is_punct(tokens[j]):
            j += 1
        if j >= n:
            continue

        tj = tokens[j]

        rhs_lem = _canon(tj.get("lemma") or tj.get("tok"))
        if _canon(ti.get("tok")) == "تلویزیون" and rhs_lem in {"نگاه","تماشا"}:
            # Peek for a following verb k
            k = j + 1
            while k < n and _is_punct(tokens[k]):
                k += 1
            if k < n and (tokens[k].get("pos","") or "").startswith("VERB"):
                # Scrub Ezafe on token i
                ti.setdefault("misc", {}).pop("Ezafe", None)
                feats = ti.get("feats", {})
                if feats.get("Case") == "Ez":  # keep other case values intact
                    feats.pop("Case", None)
                ti["feats"] = feats

        if _canon(ti.get("tok")) == "پسر" and rhs_lem == "نردبان":
            # Look ahead for 'را' and then a گرفتن-verb
            k = j + 1
            saw_ra = False
            while k < n and (_is_punct(tokens[k]) or tokens[k].get("tok") == "را"):
                if tokens[k].get("tok") == "را":
                    saw_ra = True
                k += 1
            if saw_ra and k < n and tokens[k].get("lemma") == "گرفتن" and tokens[k].get("pos","").startswith("VERB"):
                # remove Ezafe on «پسر»
                ti.setdefault("misc", {}).pop("Ezafe", None)
                feats = ti.get("feats", {})
                if feats.get("Case") == "Ez":
                    feats.pop("Case", None)
                ti["feats"] = feats

    return tokens

# --- AUDIT HARD-PATCH: Batch-only relational 'بالا' fix in N[Ez] + بالا ---
def _batch_relational_bala_final_pass(tokens: list[dict]) -> list[dict]:
    """
    In 'N/PROPN[Ez] + بالا', force 'بالا' to be NOUN (relational head).
    This is a surgical clamp applied at the very end for this batch.
    """
    n = len(tokens)
    def _is_punct(t): return (t.get("pos","") or "").startswith("PUNCT")
    def _canon(s):    return canon_chars((s or "")).replace("\u200c", "")

    for i, ti in enumerate(tokens):
        # Left token must be N/PROPN bearing Ezafe
        if not (ti.get("pos","").startswith(("NOUN","PROPN")) and ti.get("misc",{}).get("Ezafe") == "Yes"):
            continue

        # Find next non-punct token
        j = i + 1
        while j < n and _is_punct(tokens[j]):
            j += 1
        if j >= n:
            continue

        tj = tokens[j]
        if _canon(tj.get("lemma") or tj.get("tok")) == "بالا":
            # Clamp as relational NOUN
            tj["pos"]   = "NOUN"
            tj["lemma"] = "بالا"
            # Do NOT rewrite token; we are not producing 'بالای ...' here.
    return tokens

def _batch_fix_mizgard(tokens: list[dict]) -> list[dict]:
    """
    Batch-only clamp: collapse spurious 'می‌زگرد' (or 'میزگرد') to the lexical NOUN 'میزگرد'.
    """
    for t in tokens:
        s = t.get("tok", "")
        if s in {"می‌زگرد", "میزگرد"}:  # tolerate both broken and already-collapsed inputs
            t["tok"]   = "میزگرد"
            t["lemma"] = "میزگرد"
            t["pos"]   = "NOUN"
            t["morph_segments"] = [{"form": "میزگرد", "role": "stem", "morph_pos": "N"}]
    return tokens

def _batch_force_ezafe_on_bala(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    def _canon(s: str) -> str:
        return canon_chars((s or "")).replace("\u200c","")

    for i, t in enumerate(tokens[:-1]):
        if _canon(t.get("tok")) != "بالا":
            continue

        # NEW: if any PUNCT appears before the first non-PUNCT token, bail
        j = i + 1
        saw_punct = False
        while j < n and tokens[j].get("pos") == "PUNCT":
            saw_punct = True
            j += 1
        if saw_punct or j >= n:
            continue

        nxt = tokens[j]
        nxt_pos = nxt.get("pos","")
        nxt_lem_nz = _canon(nxt.get("lemma") or nxt.get("tok"))

        # If follower is DET, require adjacency to its head (no PUNCT between)
        if nxt_pos == "DET":
            k = j + 1
            if k >= n or tokens[k].get("pos") == "PUNCT":
                continue
            nxt = tokens[k]
            nxt_pos = nxt.get("pos","")
            nxt_lem_nz = _canon(nxt.get("lemma") or nxt.get("tok"))

        # Keep your existing LVC + idiom guards
        if nxt_pos.startswith("NOUN") and (nxt.get("lemma") in LVC_NOUNS_2 or _canon(nxt.get("tok")) in LVC_NOUNS_2):
            # (existing lookahead for a following verb remains as-is)
            continue
        if nxt_lem_nz in {"سر","سرش"}:
            continue

        if nxt_pos.startswith(("NOUN","PROPN","PRON","DET","NUM","ADJ")):
            t["pos"] = "NOUN"; t["lemma"] = "بالا"
            if t.get("tok") in {"بالا","بالای"}:
                t["tok"] = "بالای"
            t.setdefault("feats", {})["Case"] = "Ez"
            t.setdefault("misc",  {})["Ezafe"] = "Yes"
            t.setdefault("lemma_src","rule_based")
            t["morph_segments"] = [{"form": t["tok"], "role": "stem", "morph_pos": "N"}]
    return tokens




def _batch_fix_haman(tokens: list[dict]) -> list[dict]:
    for t in tokens:
        if t.get("tok") == "ه\u200cمان":
            t["tok"] = "همان"
        if (t.get("lemma") or "") == "ه\u200cمان":
            t["lemma"] = "همان"
    return tokens

def _force_morphpos_from_pos(tokens: list[dict]) -> list[dict]:
    POS2MP = {"NOUN":"N","PROPN":"N","ADJ":"ADJ","ADV":"ADV","ADP":"ADP","PRON":"PRON",
              "PART":"PART","CCONJ":"CCONJ","SCONJ":"SCONJ","NUM":"NUM","AUX":"AUX","PUNCT":"PUNCT"}
    for t in tokens:
        base = (t.get("pos","") or "").split(",")[0]
        mp = POS2MP.get(base)

        if not mp: 
            continue
        for seg in t.get("morph_segments", []):
            if seg.get("role") == "stem":
                seg["morph_pos"] = "V" if base.startswith("VERB") else mp
    return tokens

def _batch_scrub_tail_m(tokens: list[dict]) -> list[dict]:
    TARGETS = {"اعم": ("ADJ","اعم"), "سیستم": ("NOUN","سیستم"),"باهم": ("ADV","باهم")}
    for t in tokens:
        surf = canon_chars(t.get("tok","") or "")
        if surf in TARGETS:
            pos_expect, lem = TARGETS[surf]
            ms = [seg for seg in t.get("morph_segments", []) if seg.get("role") != "PRON_CL"]
            t["morph_segments"] = ms
            t["had_pron_clitic"] = False
            t["lemma"] = lem
            if not t.get("pos","").startswith(pos_expect):
                t["pos"] = pos_expect
    return tokens

def _batch_fix_dar_hali(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens[:-1]):
        if t.get("tok") == "حالی" and t.get("pos","").startswith("NOUN"):
            if i > 0 and tokens[i-1].get("tok") == "در":
                # next non-PUNCT must look like a nominal head
                j = i + 1
                while j < n and tokens[j].get("pos") == "PUNCT":
                    j += 1
                if j < n and tokens[j].get("pos","").startswith(("NOUN","ADJ","PROPN","NUM","PRON")):
                    t["tok"] = "حال"
                    t["lemma"] = "حال"
                    t.setdefault("feats", {})["Case"] = "Ez"
                    t.setdefault("misc", {})["Ezafe"] = "Yes"
                    t["morph_segments"] = [{"form":"حال", "role":"stem", "morph_pos":"N"}]
    return tokens

def _retag_yfinal_nominal(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens):
        pos = t.get("pos", "")
        s   = t.get("tok", "") or ""
        if not pos.startswith("VERB"):
            continue
        if not s.endswith("ی"):               # only the -y final surface
            continue
        if s.startswith(("می", "نمی")):       # rule excludes verbal prefixes
            continue
        # safety: if this token has verbal AGR, leave it alone (true finite verb)
        if any(seg.get("role") == "AGR" for seg in t.get("morph_segments", [])):
            continue

        # look‑ahead: either directly followed by a PP head (در/تو) or
        # a copula within next 3 tokens ⇒ strong nominal reading
        pp_next = (i + 1 < n and tokens[i+1].get("tok") in {"در", "تو"})
        cop_soon = any(_is_copula(tokens[j]) for j in range(i+1, min(i+4, n)))

        if pp_next or cop_soon:
            t["pos"] = "NOUN"
            # rebuild morph to nominal
            t["morph_segments"] = [{"form": s, "role": "stem", "morph_pos": "N"}]
    return tokens

# --- NEW: Demote finite verbs (with AGR) to masdar after «در/تو حالِ …»
def _normalize_hal_finite_to_masdar(tokens: list[dict]) -> list[dict]:
    """
    After [ADP در|تو] + [NOUN حال/حالی (+Ezafe)], force the *next verb* to a masdar
    when it is finite (has AGR) or is a malformed finite like …دنن. Produce a nominal
    masdar (pos=NOUN), clear AGR, and ensure idempotency.
    """
    n = len(tokens)
    for i, t in enumerate(tokens):
        # Only verbs are candidates
        if not str(t.get("pos", "")).startswith("VERB"):
            continue

        # Check left context: [در|تو] + [حال|حالی] just before i (skipping punctuation)
        p = i - 1
        while p >= 0 and tokens[p].get("pos") == "PUNCT":
            p -= 1
        if p < 1:
            continue

        is_hal = tokens[p].get("tok") in {"حال", "حالی"} and tokens[p].get("pos", "").startswith("NOUN")
        is_pp  = tokens[p-1].get("tok") in {"در", "تو"} and tokens[p-1].get("pos") == "ADP"
        if not (is_hal and is_pp):
            continue

        s = t.get("tok") or ""
        s_nz = s.replace("\u200c", "")
        ms = t.get("morph_segments", [])

        # Finite if we have AGR; also catch malformed “…دنن” (e.g., «رفتندن»)
        has_agr   = any(seg.get("role") == "AGR" for seg in ms)
        malformed = bool(re.search(r'دنن$', s_nz))  # collapse duplicated -ن after -دن

        if not (has_agr or malformed):
            # Already a masdar or something nominal-looking → leave it
            continue

        # --- Choose a safe masdar ---
        vlem = (t.get("lemma") or "").replace("\u200c", "")

        def _best_masdar(surface: str, lemma: str) -> str:
            # If lemma already looks like a masdar (…دن/…تن/…شن), trust it
            if re.search(r'(?:دن|تن|شن)$', lemma):
                return lemma
            # Otherwise, extract finite morphology from the surface and build stem+ن
            parts = _extract_verb_affixes(surface)
            stem = parts.get("stem") or lemma or surface
            # Safety: trim any stray AGR stuck on the stem
            stem = re.sub(r'(?:م|ی|د|یم|ید|ند|اند)$', '', stem)
            out = stem + "ن"
            # Normalize weird double-n: …دنن → …دن
            out = re.sub(r'دنن$', 'دن', out)
            return out

        masdar = _best_masdar(s, vlem)

        # --- Write back as a nominal masdar (idempotent) ---
        t["tok"] = masdar
        t["lemma"] = masdar
        t["pos"] = "NOUN"
        t["morph_segments"] = [{"form": masdar, "role": "stem", "morph_pos": "N"}]
        t["had_pron_clitic"] = False
    return tokens

def _batch_force_masdar_after_hal(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens[:-1]):
        if t.get("tok") != "حال":
            continue
        # look left for 'در' or 'تو' without crossing PUNCT/PART
        p = i - 1
        while p >= 0 and tokens[p].get("pos") in {"PUNCT","PART"}:
            p -= 1
        if p < 0 or tokens[p].get("tok") not in {"در","تو"}:
            continue

        # first non-PUNCT to the right
        j = i + 1
        while j < n and tokens[j].get("pos") == "PUNCT":
            j += 1
        if j >= n:
            continue

        v = tokens[j]
        if v.get("pos","").startswith("VERB") and v.get("lemma") == "رفتن" and v.get("tok","").endswith("ند"):
            # rewrite to masdar and drop AGR
            v["tok"] = "رفتن"
            v["morph_segments"] = [seg for seg in v.get("morph_segments", []) if seg.get("role") != "AGR"]
    return tokens

def _batch_fix_yad_typo(tokens: list[dict]) -> list[dict]:
    for t in tokens:
        if canon_chars(t.get("tok","")) == "یعد":
            t["tok"] = "بعد"
            if canon_chars(t.get("lemma","")) == "یعد":
                t["lemma"] = "بعد"
            # keep POS as-is; refresh stem form if present
            for seg in t.get("morph_segments", []):
                if seg.get("role") == "stem":
                    seg["form"] = "بعد"
    return tokens


def _fix_raftoamad_spacing(text: str) -> str:
    """
    Split only the idiom 'در رفت و آمد...' when it appears fused as 'دررفت و آمد…'.
    Preserve any verbal AGR/clitic on 'آمد…' (e.g., آمدند، آمدندش).
    Idempotent: won't re-touch already-spaced forms.
    """
    import re
    agr  = r'(?:م|ی|د|یم|ید|ند|اند)?'  # finite past AGR endings
    clt  = r'(?:\u200c)?(?:ام|ات|اش|مان|تان|شان|مون|تون|شون|م|ت|ش)?'  # optional clitic
    punct = r'[؟\.,،؛!»\)\]]'

    pat = re.compile(
        rf'(?<!\S)'                    # token start
        r'در' r'[\u200c\-]?\s*'        # fused 'در' (+ optional joiner)
        r'رفت' r'\s*'                  # 'رفت'
        r'و'   r'\s*'                  # 'و'
        r'آمد' f'{agr}{clt}'           # 'آمد' + AGR/clitic (optional)
        rf'(?=\s|{punct}|$)'           # do not overrun
    )

    def _repl(m: re.Match) -> str:
        s = m.group(0)
        # keep whatever trails after 'آمد' (AGR/clitic)
        tail = re.sub(r'.*آمد', '', s, flags=re.DOTALL)
        return 'در رفت و آمد' + tail

    return pat.sub(_repl, text)


_INSHALLAH_RX = re.compile(
    r'(?<!\S)(?:ان\s*شا?ء?\s*الله|انشاالله|انشالله)(?!\S)'
)

def _normalize_inshallah(text: str) -> str:
    # unify to «ان‌شاءالله» with ZWNJ between «ان» and «شاءالله»
    return _INSHALLAH_RX.sub("ان" + _ZWNJ + "شاءالله", text)

def _scrub_sentence_final_bala(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t.get("tok") != "بالای":
            continue
        # find first non-PUNCT follower
        j = i + 1
        while j < n and tokens[j].get("pos") == "PUNCT":
            j += 1
        if j >= n:
            # sentence-final: strip Ezafe
            t["tok"] = "بالا"
            t.setdefault("feats", {}).pop("Case", None)
            t.setdefault("misc", {}).pop("Ezafe", None)
    return tokens

def _normalize_ruberoo_adp(tokens: list[dict]) -> list[dict]:
    """
    Normalize relational «روبه‌رو/روبرو» (with or without possessive clitic)
    to ADP head «روبه‌روی(+clitic)» + Ezafe when it governs a nominal head.
    """
    n = len(tokens)

    def _canon(s: str) -> str:
        return canon_chars((s or "")).replace(_ZWNJ, "")

    HEAD_BASES = {"روبهرو", "روبرو", "روبه‌رو", "روبروی"}  # tolerate both spellings

    for i, t in enumerate(tokens):
        s = t.get("tok", "") or ""
        pos = t.get("pos", "")

        if pos not in {"ADP", "ADV", "NOUN"}:
            continue

        # peel a possessive clitic (if any) before matching the base
        core, pron = _match_longest_suffix(s, tuple(_PRON_MAP.keys()))
        core_nz = _canon(core)

        if core_nz not in HEAD_BASES and core_nz != "روبروی":
            continue

        # find first non-PUNCT follower; require a nominal-like complement
        j = i + 1
        saw_punct = False
        while j < n and tokens[j].get("pos") == "PUNCT":
            saw_punct = True
            j += 1
        if saw_punct or j >= n:
            continue

        nxt_pos = tokens[j].get("pos", "")
        if nxt_pos == "DET":
            k = j + 1
            if k >= n or tokens[k].get("pos") == "PUNCT":
                continue
            nxt_pos = tokens[k].get("pos", "")

        if not nxt_pos.startswith(("NOUN", "PROPN", "ADJ", "NUM", "PRON")):
            continue

        # perform normalization; preserve clitic if present
        base = "روبه‌روی"
        t["tok"]   = base + (pron or "")
        t["lemma"] = "روبه‌رو"
        t["pos"]   = "ADP"
        t.setdefault("feats", {})["Case"] = "Ez"
        t.setdefault("misc",  {})["Ezafe"] = "Yes"
        t.setdefault("lemma_src", "rule_based")
    return tokens



def _enforce_progressive_aux_postud(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens):
        if t.get("lemma") == "داشتن" and not t.get("pos","").startswith("AUX"):
            # scan next ~6 tokens; stop at clear clause boundary
            for j in range(i+1, min(i+7, n)):
                posj = tokens[j].get("pos","")
                tokj = tokens[j].get("tok","")
                if posj.startswith("VERB") and tokj.startswith("می"):
                    t["pos"] = "AUX"
                    break
                if posj in {"SCONJ","CCONJ"}:
                    break
                if posj == "PUNCT" and tokj in {".","؟","!","؛"}:
                    break
    return tokens

def _strip_balay_before_punct(tokens: list[dict]) -> list[dict]:
    for i in range(len(tokens) - 1):
        if tokens[i].get("tok") == "بالای" and tokens[i+1].get("pos") == "PUNCT":
            tokens[i]["tok"] = "بالا"
            tokens[i].setdefault("feats", {}).pop("Case", None)
            tokens[i].setdefault("misc", {}).pop("Ezafe", None)
            if tokens[i].get("pos","").startswith(("NOUN","PROPN")):
                tokens[i]["morph_segments"] = [{"form":"بالا","role":"stem","morph_pos":"N"}]
    return tokens

def _retag_no_particle(tokens: list[dict]) -> list[dict]:
    """
    Conservative, idempotent retagger for «نه».
    - Stand-alone “نه” (utterance-final/initial or wrapped by PUNCT) → INTJ
    - Contrastive “نه” within clauses (e.g., «…، نه X …») → PART
    Leaves pre-verbal negation morphology (نـ) untouched (that is handled elsewhere).
    """
    n = len(tokens)
    for i, t in enumerate(tokens):
        if (t.get("tok") or "") != "نه":
            continue
        prev = tokens[i-1] if i > 0 else None
        nxt  = tokens[i+1] if i+1 < n else None
        prev_punct = (prev is None) or prev.get("pos") == "PUNCT"
        next_punct = (nxt  is None) or nxt.get("pos")  == "PUNCT"
        if prev_punct and next_punct:
            t["pos"] = "INTJ"  # stand-alone “no”
        elif prev_punct or next_punct:
            t["pos"] = "PART"  # contrastive edge marker
        else:
            # Default to particle in mid-clause contrast
            t["pos"] = "PART"
    return tokens
# =============================== FINAL-PASS REPAIRS ===============================
def _final_idempotent_repairs(tokens: list[dict]) -> list[dict]:
    """Final, conservative, idempotent repairs:
       (i) enforce AUX copula AGR split (… + «ند») even if manual-map bypassed it,
       (ii) re-tag bare «نه» (ADV residues) to PART/INTJ by punctuation-bounded context,
       (iii) add missing PRON_CL on ZWNJ compounds (… + «مون/تون/شون/مان/تان/شان»).
    """
    tokens = _force_aux_copula_agr_split(tokens)
    tokens = _retag_no_backup(tokens)
    tokens = _repair_pron_clitic_on_zwnj_compounds(tokens)
    return tokens


def _force_aux_copula_agr_split(tokens: list[dict]) -> list[dict]:
    """If a finite AUX copula ends in «ند» and has no AGR segment, split into stem+AGR.
       Covers forms like «هستند», «نیستند», and other AUX=بودن 3Pl realisations.
       Idempotent: does nothing if an AGR segment already exists.
    """
    ZWNJ = "\u200c"
    for t in tokens:
        pos   = t.get("pos", "")
        tok   = t.get("tok", "") or ""
        lemma = (t.get("lemma") or "")
        segs  = t.get("morph_segments") or []
        if not pos.startswith("AUX"):
            continue

        # Accept ...«ند» and ZWNJ-separated «‌ند»
        ends_plain = tok.endswith("ند")
        ends_zwnj  = tok.endswith(ZWNJ + "ند")
        if not (ends_plain or ends_zwnj):
            continue

        # Already split? — if any AGR exists, skip (idempotent)
        agr_idx = next((i for i, s in enumerate(segs) if s.get("role") == "AGR"), None)
        if agr_idx is not None:
            if segs[agr_idx].get("form") == "د" and (ends_plain or ends_zwnj):
                segs[agr_idx]["form"]   = "ند"
                segs[agr_idx]["Person"] = "3"
                segs[agr_idx]["Number"] = "Plur"
                # trim a trailing «ن» from the stem if present → …ست + «ند»
                stem_idx = next((i for i in range(len(segs)-1, -1, -1) if segs[i].get("role") == "stem"), None)
                if stem_idx is not None and (segs[stem_idx].get("form") or "").endswith("ن"):
                    segs[stem_idx]["form"] = segs[stem_idx]["form"][:-1]
                t["morph_segments"] = segs
            continue

        # Restrict to copular AUXes; your data lemmas for copula are typically «بودن».
        # Some corpora set lemma=«نیست» for negated copula; allow both safely.
        if lemma not in {"بودن", "نیست", "باشد"} and tok not in {"هستند", "نیستند"}:
            # Keep conservative scope: only the copula, not arbitrary AUX tokens.
            continue

        # Compute base and AGR forms
        if ends_zwnj:
            base = tok[:-(len("ند")+1)]
            agr  = "ند"
        else:
            base = tok[:-2]
            agr  = "ند"

        # Rebuild morph_segments deterministically
        new_segs = [{"form": base, "role": "stem", "morph_pos": "AUX"},
                    {"form": agr,  "role": "AGR",  "Person": "3", "Number": "Plur"}]
        t["morph_segments"] = new_segs
        # Keep token string/lemma/pos as-is; UD features can be derived downstream.
    return tokens


def _retag_no_backup(tokens: list[dict]) -> list[dict]:
    """Backup retagger for bare «نه»:
       - Stand-alone (bounded by sentence or PUNCT on both sides) → INTJ
       - Clause-edge/contrast (one side punctuated) → PART
       - Otherwise mid-clause contrast → PART
       Only overwrites residual ADV tags; leaves other tags intact.
       Idempotent and narrow-scoped.
    """
    n = len(tokens)
    for i, t in enumerate(tokens):
        if (t.get("tok") or "") != "نه":
            continue
        if t.get("pos") != "ADV":
            # If earlier passes already set PART/INTJ correctly, do nothing.
            continue
        prev = tokens[i-1] if i > 0 else None
        nxt  = tokens[i+1] if i+1 < n else None
        prev_punct = (prev is None) or (prev.get("pos") == "PUNCT")
        next_punct = (nxt  is None) or (nxt.get("pos")  == "PUNCT")
        if prev_punct and next_punct:
            t["pos"] = "INTJ"
        else:
            t["pos"] = "PART"
    return tokens


def _repair_pron_clitic_on_zwnj_compounds(tokens: list[dict]) -> list[dict]:
    """Repair still-fused possessive PRON clitics on ZWNJ compounds (e.g., «هفت‌سینمون»).
       We only act when:
         • token contains a ZWNJ somewhere (compound), AND
         • token ends with one of the multi-letter clitic strings, AND
         • no PRON_CL segment is present yet (idempotent), AND
         • coarse POS is NOUN/PROPN/ADJ (typical hosts for possessive clitics).
       This avoids lexical false alarms like «نشان», because they lack ZWNJ.
    """
    ZWNJ = "\u200c"

    # map clitic surface → (Person, Number)
    CL = {
        "مون": ("1", "Plur"),
        "تون": ("2", "Plur"),
        "شون": ("3", "Plur"),
        "مان": ("1", "Plur"),
        "تان": ("2", "Plur"),
        "شان": ("3", "Plur"),
    }

    for t in tokens:
        pos = t.get("pos", "")
        if pos not in {"NOUN", "PROPN", "ADJ"}:
            continue
        s   = (t.get("tok") or "")
        if "\u200c" not in s:
            continue  # only compounds; prevents false positives like «نشان»
        segs = t.get("morph_segments") or []
        if any(seg.get("role") == "PRON_CL" for seg in segs) or t.get("had_pron_clitic"):
            continue  # already analysed — idempotent

        # Try each clitic; prefer the longest match (multi-letter only)
        matched = None
        for suff in ("شون", "مون", "تون", "شان", "مان", "تان"):
            if s.endswith(suff):
                matched = suff
                break
        if not matched:
            continue

        stem = s[: -len(matched)]
        person, number = CL[matched]

        # Update morph_segments: replace a monolithic stem with split stem + PRON_CL
        if len(segs) == 1 and segs[0].get("role") == "stem":
            segs = [{"form": stem, "role": "stem", "morph_pos": segs[0].get("morph_pos", "N")},
                    {"form": matched, "role": "PRON_CL",
                     "Person": person, "Number": number, "Case": "Poss"}]
        else:
            # Append conservatively
            segs.append({"form": matched, "role": "PRON_CL",
                         "Person": person, "Number": number, "Case": "Poss"})
            # If the last stem segment still shows the fused surface, fix its form
            for seg in segs:
                if seg.get("role") == "stem" and seg.get("form") == s:
                    seg["form"] = stem
                    break

        t["morph_segments"] = segs
        t["had_pron_clitic"] = True
    return tokens
# ==================
# =============================== BATCH‑2 FINAL PATCH ===============================

def _final_batch2_repairs(tokens: list[dict]) -> list[dict]:
    """Idempotent, narrow-scope repairs for Batch‑2:
       (i) Fix 3SG AGR (…ن + د) vs. 3PL AGR (… + ند) in finite VERB forms,
       (ii) Retag masdar after «مشغول … [NP] + مصدر» to VERB(Inf),
       (iii) Ensure Ezafe flags when _had_ez is true on the nominal host.
    """
    tokens = _fix_fin_verb_agr_3sg(tokens)
    tokens = _retag_mashghul_np_masdar(tokens)
    tokens = _ensure_ezafe_feature_consistency(tokens)
    return tokens


def _fix_fin_verb_agr_3sg(tokens: list[dict]) -> list[dict]:
    """
    If a finite VERB was segmented with AGR='ند' but the *stem* already ends in «ن»
    and the *surface* does NOT end in «نند», then it's the classic 3SG case:
        stem(…ن) + «د»  ⇒ surface …ند   (e.g., «می‌بیند», «می‌خواند», «کند»)
    We correct AGR to «د» (Person=3, Number=Sing) and, if needed, append «ن» to the stem.
    We NEVER touch cases ending in «نند» (true 3PL: «می‌بینند», «کنند») or stems not ending in «ن».
    Idempotent: does nothing when AGR is already «د» or the conditions fail.
    """
    for t in tokens:
        if not (t.get("pos", "").startswith("VERB")):
            continue
        segs = t.get("morph_segments") or []
        if not segs:
            continue

        # Find one AGR segment
        agr_idx = next((i for i, s in enumerate(segs) if s.get("role") == "AGR"), None)
        if agr_idx is None:
            continue
        agr = segs[agr_idx]
        if agr.get("form") != "ند":
            continue  # only reconsider mis-labeled plural AGR

        tok = (t.get("tok") or "")
        if tok.endswith("نند"):
            # unequivocally true 3PL (… + «ند» after a stem ending in «ن»)
            continue

        # Lookup last stem segment
        stem_idx = None
        for i in range(len(segs)-1, -1, -1):
            if segs[i].get("role") == "stem":
                stem_idx = i
                break
        if stem_idx is None:
            continue
        stem = segs[stem_idx]
        stem_form = stem.get("form") or ""

        # We only flip to 3SG when the *stem already ends in ن*:
        # This protects true 3PL forms like «گویند/گیرند/گویند…» (stems not ending in «ن»).
        if not stem_form.endswith("ن"):
            continue

        # At this point it's the ambiguous 3SG surface «…ند» = stem(…ن) + «د».
        # Normalize AGR to 3SG and keep the stem intact (already ends with «ن»).
        agr.update({"form": "د", "Person": "3", "Number": "Sing"})
        # Idempotent: if already fixed, values remain the same.
    return tokens


def _retag_mashghul_np_masdar(tokens: list[dict]) -> list[dict]:
    """
    Pattern: «مشغول … [NP up to 2 tokens] + MASDAR»  ⇒ make MASDAR a VERB(Inf).
    - We allow at most two light NP tokens in between (DET/ADP/ADJ/NOUN/PRON).
    - The MASDAR candidate must end with «ن» and have no AGR segment.
    - If candidate is already VERB(Inf), we leave it.
    Idempotent, very narrow scope (only when 'مشغول' appears).
    """
    permissive_gap_pos = {"DET", "ADP", "ADJ", "NOUN", "PRON"}
    for i, t in enumerate(tokens):
        if (t.get("tok") != "مشغول") or (t.get("pos") not in {"ADJ", "NOUN"}):
            continue

        # Scan next tokens with a small window and skip permissible NP material
        j = i + 1
        steps = 0
        while j < len(tokens) and steps < 3:
            cand = tokens[j]
            if cand.get("pos") in {"PUNCT", "CCONJ", "SCONJ", "PART", "AUX"}:
                break  # clause boundary or functional material -> stop
            if cand.get("pos") in permissive_gap_pos:
                # check if this 'gap' item is itself the masdar
                if _looks_like_masdar(cand) and not _has_agr(cand):
                    _force_infinitive_pos(cand)
                else:
                    # keep scanning up to 2 such tokens
                    steps += 1
                    j += 1
                    continue
            else:
                # Not permissive POS; if it looks like a masdar, fix; else stop.
                if _looks_like_masdar(cand) and not _has_agr(cand):
                    _force_infinitive_pos(cand)
                break
            j += 1
    return tokens


def _looks_like_masdar(tok: dict) -> bool:
    """Heuristic for Persian masdar form: ends with «ن», no prior AGR."""
    s = (tok.get("tok") or "")
    return s.endswith("ن")


def _has_agr(tok: dict) -> bool:
    for s in (tok.get("morph_segments") or []):
        if s.get("role") == "AGR":
            return True
    return False


def _force_infinitive_pos(tok: dict) -> None:
    # If already VERB(Inf), leave as is
    if tok.get("pos") == "VERB" and tok.get("feats", {}).get("VerbForm") == "Inf":
        return
    tok["pos"] = "VERB"
    feats = tok.setdefault("feats", {})
    feats["VerbForm"] = "Inf"
    # Promote stem's morph_pos to V where available
    for s in tok.get("morph_segments") or []:
        if s.get("role") == "stem":
            s["morph_pos"] = "V"


def _ensure_ezafe_feature_consistency(tokens: list[dict]) -> list[dict]:
    """
    When tokenizer marked a token with '_had_ez'=True (host of Ezafe),
    enforce feats.Case='Ez' and misc.Ezafe='Yes' on *nominal* hosts.
    We do NOT add Ezafe to ADP/VERB/AUX/PUNCT/etc. to avoid regressions.
    Idempotent: only fills missing fields.
    """
    blocked = {"ADP", "VERB", "AUX", "PUNCT", "CCONJ", "SCONJ", "PART"}
    n = len(tokens)
    for i, t in enumerate(tokens):
        if not t.get("_had_ez"):
            continue
        if t.get("pos") in blocked:
            continue

        if i > 0 and tokens[i-1].get("pos") == "ADP":
            j = i + 1
            # skip punctuation
            while j < n and tokens[j].get("pos") == "PUNCT":
                j += 1
            follows_dep = (j < n) and tokens[j].get("pos","").startswith(("ADJ","DET","PRON","NUM","PROPN","NOUN"))
            special_case = (tokens[i-1].get("tok") == "در" and t.get("tok") in {"مورد","حال"})
            if not (follows_dep or special_case):
                continue
        # (2) Predicative guard: N* + ADJ … + COP within a short window
        j = i + 1
        # skip over punctuation
        while j < n and tokens[j].get("pos") == "PUNCT":
            j += 1
        if j < n and tokens[j].get("pos","").startswith("ADJ"):
            k = j + 1
            # allow intervening ADJ/NOUN/NUM up to the copula
            hops = 0
            while k < n and hops < 5 and tokens[k].get("pos","") in {"ADJ","NOUN","NUM","PROPN","PUNCT"}:
                if tokens[k].get("pos") == "PUNCT":
                    hops += 1
                k += 1
            if k < n and _is_copula(tokens[k]):
                continue     

        feats = t.setdefault("feats", {})
        misc  = t.setdefault("misc", {})
        if feats.get("Case") != "Ez":
            feats["Case"] = "Ez"
        if misc.get("Ezafe") != "Yes":
            misc["Ezafe"] = "Yes"

    return tokens

# ============================ END BATCH‑2 FINAL PATCH ============================

# ============================ BATCH‑2 HARD PATCH (IDEMPOTENT) ============================

def _final_batch2_hard_patch(tokens: list[dict]) -> list[dict]:
    """Idempotent and narrow-scoped repairs:
       (i) Fix ambiguous 3SG vs 3PL AGR on finite verbs when the stem ends with «ن»,
       (ii) Promote MASDAR to VERB(Inf) in the phase frames:
            «مشغول … [≤2 NP tokens] … MASDAR … است» and «در حال … MASDAR … است»,
       (iii) Materialise Ezafe features if _had_ez is True on nominal heads,
             but never in predicative N ADJ COP configurations.
    """
    tokens = _fix_agr_3sg_when_stem_ends_in_noon(tokens)
    tokens = _phase_frame_infinitive_upgrader(tokens)
    tokens = _project_ezafe_from_flag_nominal_safe(tokens)
    return tokens


# --------------------------- (i) AGR 3SG vs 3PL disambiguation ---------------------------

def _fix_agr_3sg_when_stem_ends_in_noon(tokens: list[dict]) -> list[dict]:
    """
    If a finite VERB has AGR segment «ند» but:
      • the *surface* token does NOT end in «نند» (true -ند plural), and
      • the last 'stem' segment form ends with «ن»,
    then this is the classical 3SG case (stem(…ن)+«د») – fix AGR to «د» with 3SG features.
    Idempotent: no change if AGR already «د» or if the guards fail.
    """
    for t in tokens:
        if not (t.get("pos", "").startswith("VERB")):
            continue
        segs = t.get("morph_segments") or []
        if not segs:
            continue

        # Find AGR
        agr_idx = next((i for i, s in enumerate(segs) if s.get("role") == "AGR"), None)
        if agr_idx is None or segs[agr_idx].get("form") != "ند":
            continue  # only reconsider suspicious plural AGR

        tok = (t.get("tok") or "")
        if tok.endswith("نند"):
            continue  # unambiguously 3PL: keep «ند»

        # locate last stem segment
        stem_idx = None
        for i in range(len(segs) - 1, -1, -1):
            if segs[i].get("role") == "stem":
                stem_idx = i
                break
        if stem_idx is None:
            continue

        stem_form = (segs[stem_idx].get("form") or "")
        if not stem_form.endswith("ن"):
            continue  # stems not ending with «ن» (e.g., «گوی/گیر») → plural truly «ند»

        # Flip AGR to 3SG «د»
        segs[agr_idx].update({"form": "د", "Person": "3", "Number": "Sing"})
        t["morph_segments"] = segs  # (idempotent if already corrected)
    return tokens


# --------------------------- (ii) Phase-frame infinitive upgrader ------------------------

def _phase_frame_infinitive_upgrader(tokens: list[dict]) -> list[dict]:
    """
    Promote MASDAR to VERB(VerbForm=Inf) in two frames:
      A) «مشغول … [≤2 NP tokens] … MASDAR … است»
      B) «در حال … MASDAR … است»
    We require:
      • candidate ends with «ن» (surface heuristic for Persian infinitive),
      • candidate has NO AGR segment,
      • an AUX copula appears within the next 4 tokens after the candidate,
      • for (A): allow up to two 'permissible NP' tokens between anchor and candidate.
    Idempotent: leaves already VERB(Inf) tokens unchanged.
    """
    # permissive 'gap' POS that may appear between anchor and masdar
    NP_OK = {"DET", "ADP", "ADJ", "NOUN", "PROPN", "PRON", "NUM"}

    n = len(tokens)
    i = 0
    while i < n:
        t = tokens[i]

        # ---- Frame A: «مشغول … MASDAR … است»
        if t.get("tok") == "مشغول" and t.get("pos") in {"ADJ", "NOUN"}:
            j = i + 1
            hops = 0
            while j < n and hops <= 2:
                cand = tokens[j]
                if cand.get("pos") in {"PUNCT", "SCONJ", "CCONJ"}:
                    break
                if _looks_like_masdar(cand) and not _has_agr(cand) and _has_aux_within(tokens, j, max_ahead=4):
                    _force_infinitive_pos(cand)
                    break
                if cand.get("pos") in NP_OK:
                    hops += 1
                    j += 1
                    continue
                break  # unexpected material
            i += 1
            continue

        # ---- Frame B: «در حال … MASDAR … است»
        if t.get("tok") == "در" and t.get("pos") == "ADP":
            # expect «حال» right after (optionally with Ezafe)
            if i + 1 < n and tokens[i + 1].get("tok") == "حال":
                j = i + 2  # scan after «حال»
                # allow at most one NP token (e.g., a light noun before masdar), then candidate
                hops = 0
                while j < n and hops <= 1:
                    cand = tokens[j]
                    if cand.get("pos") in {"PUNCT", "SCONJ", "CCONJ"}:
                        break
                    if _looks_like_masdar(cand) and not _has_agr(cand) and _has_aux_within(tokens, j, max_ahead=4):
                        _force_infinitive_pos(cand)
                        break
                    if cand.get("pos") in NP_OK:
                        hops += 1
                        j += 1
                        continue
                    break
                i += 2
                continue

        i += 1

    return tokens


def _looks_like_masdar(tok: dict) -> bool:
    """Conservative masdar test: surface ends with «ن» and POS is not PRON/ADP/AUX."""
    s = (tok.get("tok") or "")
    if not s.endswith("ن"):
        return False
    if tok.get("pos") in {"PRON", "ADP", "AUX"}:
        return False
    return True


def _has_agr(tok: dict) -> bool:
    """True if any morph segment is AGR."""
    return any(seg.get("role") == "AGR" for seg in (tok.get("morph_segments") or []))


def _has_aux_within(tokens: list[dict], start_idx: int, max_ahead: int = 4) -> bool:
    """Check if an AUX appears within the next k tokens (punct allowed)."""
    end = min(len(tokens), start_idx + 1 + max_ahead)
    for k in range(start_idx + 1, end):
        if tokens[k].get("pos") == "AUX":
            return True
        if tokens[k].get("pos") == "PUNCT":
            # tolerate commas/periods: keep scanning
            continue
    return False


def _force_infinitive_pos(tok: dict) -> None:
    """Make token VERB with VerbForm=Inf; keep segments idempotently."""
    if tok.get("pos") == "VERB" and tok.get("feats", {}).get("VerbForm") == "Inf":
        return
    tok["pos"] = "VERB"
    feats = tok.setdefault("feats", {})
    feats["VerbForm"] = "Inf"
    for s in tok.get("morph_segments") or []:
        if s.get("role") == "stem":
            s["morph_pos"] = "V"


# --------------------------- (iii) Ezafe projection with predicative guard --------------

def _project_ezafe_from_flag_nominal_safe(tokens: list[dict]) -> list[dict]:
    """
    If tokenizer marked a token with _had_ez=True (host of Ezafe), enforce:
        feats.Case="Ez" and misc.Ezafe="Yes"
    but ONLY when the host is nominal (NOUN/PROPN/ADJ/NUM) and NOT in a
    predicative N ADJ COP configuration within a short window.
    Idempotent: fills missing fields, does not overwrite existing values.
    """
    NOMINAL = {"NOUN", "PROPN", "ADJ", "NUM"}
    BLOCK   = {"ADP", "VERB", "AUX", "PUNCT", "CCONJ", "SCONJ", "PART"}

    n = len(tokens)
    for i, t in enumerate(tokens):
        if not t.get("_had_ez"):
            continue
        if t.get("pos") not in NOMINAL:
            continue

        # Predicative guard: N + ADJ + (AUX within next 2 tokens) → likely predicate: DO NOT add Ezafe
        j = _next_non_punct(tokens, i + 1)
        if j is not None and tokens[j].get("pos") == "ADJ":
            # If an AUX appears before any intervening NOUN/PROPN/ADP, treat as predicative
            k = j + 1
            predicative = False
            while k < min(n, j + 4):
                posk = tokens[k].get("pos")
                if posk == "AUX":
                    predicative = True
                    break
                if posk in {"NOUN", "PROPN", "ADP"}:
                    break
                if posk == "PUNCT":
                    break
                k += 1
            if predicative:
                continue  # skip Ezafe projection in N ADJ COP

        # Do not add Ezafe to clearly non-nominal/bound forms
        if t.get("pos") in BLOCK:
            continue

        feats = t.setdefault("feats", {})
        misc  = t.setdefault("misc", {})
        if feats.get("Case") != "Ez":
            feats["Case"] = "Ez"
        if misc.get("Ezafe") != "Yes":
            misc["Ezafe"] = "Yes"
    return tokens


def _next_non_punct(tokens: list[dict], idx: int) -> int | None:
    """Return index of the next non-PUNCT token at/after idx; else None."""
    while 0 <= idx < len(tokens):
        if tokens[idx].get("pos") != "PUNCT":
            return idx
        idx += 1
    return None

# ======================== END BATCH‑2 HARD PATCH (IDEMPOTENT) ============================
def _normalize_lab_adp(tokens: list[dict]) -> list[dict]:
    """
    Normalize relational «لب» functioning as a head to NOUN + Ezafe when it governs an NP.
    Bail if punctuation intervenes. Idempotent and conservative.
    """
    n = len(tokens)
    for i, t in enumerate(tokens[:-1]):
        if t.get("tok") == "لب" and t.get("pos") in {"ADP","NOUN"}:
            j = i + 1
            # skip through PUNCT
            while j < n and tokens[j].get("pos") == "PUNCT":
                j += 1
            if j < n and tokens[j].get("pos","").startswith(("NOUN","ADJ","PROPN","PRON","NUM")):
                t["pos"] = "NOUN"
                t["lemma"] = "لب"
                t.setdefault("feats", {})["Case"] = "Ez"
                t.setdefault("misc", {})["Ezafe"] = "Yes"
                t.setdefault("lemma_src","rule_based")
                t["morph_segments"] = [{"form": "لب", "role": "stem", "morph_pos": "N"}]
    return tokens




def _normalize_bare_heh_copula(tokens: list[dict]) -> list[dict]:
    """
    Convert bare «هه» to finite copula «است» only in predicative contexts:
    previous token must be nominal/adj/proper/pron/num.
    """
    for i, t in enumerate(tokens):
        if t.get("tok") == "هه" and i > 0 and str(tokens[i-1].get("pos","")).startswith(("NOUN","PROPN","ADJ","PRON","NUM")):
            t["tok"] = "است"
            t["lemma"] = "بودن"
            t["pos"] = "AUX"
            t["morph_segments"] = [{"form": "است", "role": "stem", "morph_pos": "AUX"}]
    return tokens

def _canon(s: str) -> str:
    # canonical form without ZWNJ (fits your examples)
    return canon_chars(s or "").replace(_ZWNJ, "")

def _tok(tokens, i):
    return tokens[i].get("tok") if 0 <= i < len(tokens) else None

def _pos(tokens, i):
    return tokens[i].get("pos") if 0 <= i < len(tokens) else None

def _lemma(tokens, i):
    return tokens[i].get("lemma") if 0 <= i < len(tokens) else None

def _mark_ez(tok: dict) -> None:
    tok.setdefault("feats", {})["Case"] = "Ez"
    tok.setdefault("misc",  {})["Ezafe"] = "Yes"
    tok["_had_ez"] = True  # optional: keeps it consistent with the rest of the pipeline

def _unmark_ez(tok: dict) -> None:
    tok.setdefault("feats", {}).pop("Case", None)
    tok.setdefault("misc",  {}).pop("Ezafe", None)
    tok["_had_ez"] = False

# --- Your simple rule bucket: write plain if-statements here ---
def _apply_manual_ezafe_overrides(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens):
        # EXAMPLE 1 (yours): صنایعِ دستی بافتنی باشد  → put Ezafe on «دستی»
        #   if _canon(tok_i) == "دستی" and tokens[i+1].get("tok") == "بافتنی" and tokens[i+2].get("tok") == "باشد": _mark_ez(t); continue
        if (
            _canon(_tok(tokens, i)) == "دستی"
            and _tok(tokens, i+1) == "بافتنی"
            and _tok(tokens, i+2) == "باشد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پدر"
            and _tok(tokens, i+1) == "مادرش"
            and _tok(tokens, i+2) == "هستند"
            and _tok(tokens, i-1) == "احتمالا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i+2) == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "چوبی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تنگ"
            and _tok(tokens, i+1) == "دست"
            and _tok(tokens, i+2) == "او"
        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "قفسه"
            and _tok(tokens, i+1) == "کتاب"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "پشتش"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بادبادک"
            and _tok(tokens, i+1) == "هوا"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "علفی"
            and _tok(tokens, i+1) == "آفتاب"
            and _tok(tokens, i+2) == "زده"
            and _tok(tokens, i+3) == "زرد"
            and _tok(tokens, i+4) == "شده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جا"
            and _tok(tokens, i+1) == "علفی"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+2) == "آفتاب"
            and _tok(tokens, i+3) == "زده"
            and _tok(tokens, i+4) == "زرد"
            and _tok(tokens, i+5) == "شده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i+1) in {"قایق‌سواری","قایقسواری"}
            and _tok(tokens, i+2) == "دارد"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"دفترچه‌ای","دفترچهای"}
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "هوا"
            and _tok(tokens, i-1) == "روی"
            and _tok(tokens, i+1) == "توپ"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "بگیرد"
        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i-1) == "دارد"
            and _tok(tokens, i+1) == "چیزش"
            and _tok(tokens, i+2) == "دارد"
            and _tok(tokens, i+3) == "قالی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ساعتی"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "سرشان"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خارج"
            and _tok(tokens, i-1) == "باید"
            and _tok(tokens, i+1) == "از"
            and _tok(tokens, i+2) == "منزل"
            and _tok(tokens, i+3) == "باشد"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "طوفانی"
            and _tok(tokens, i+1) == "بوده"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "راجع"
            and _tok(tokens, i+1) == "به"
            and _tok(tokens, i+2) == "چه"
            and _tok(tokens, i+3) == "هست"
            and _tok(tokens, i-1) == "کتاب"
        ):
            _set_pos(t,"ADP")
            continue

        

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i-2) == "کنار"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "شانه‌اش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنترل"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "و"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "طور"
            and _tok(tokens, i+3) == "چیزها"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "و"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "آقا"
            and _tok(tokens, i+3) == "هم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"پسربچه","پسر‌بچه"}
            and _tok(tokens, i+1) == "بادبادک"
            and _tok(tokens, i+2) == "هوا"
            and _tok(tokens, i+3) == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بادبادک"
            and _tok(tokens, i+1) == "هوا"
            and _tok(tokens, i+2) == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جا"
            and _tok(tokens, i-1) == "همه"
            and _tok(tokens, i+1) == "میوه‌های"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"میوه‌های","میوههای"}
            and _tok(tokens, i+1) == "مثل"
            and _tok(tokens, i+2) == "شمالی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "همه"
            and _tok(tokens, i+1) == "آنها"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فضای"
            and _tok(tokens, i+1) == "باز"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سمت"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i-1) == "به"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرف","آنطرف"}
            and _tok(tokens, i+1) == "سه‌تا"
            and _tok(tokens, i+2) == "یک"
        ):
            _unmark_ez(t)
            continue

        

        if (
            _canon(_lemma(tokens, i)) == "عصرانه"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "نظیر"
        ):
            _unmark_ez(t)
            continue
        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1) == "نردبان"
            and _tok(tokens, i+2) == "،"
        ):
            _mark_ez(t)
            continue
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "نردبان"
            and _tok(tokens, i+2) == "،"
        ):
            _mark_ez(t)
            continue
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "درخت"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+2) == "شاخه"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "درخت"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "هوا"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "که"
            and _tok(tokens, i+3) == "نشسته"

        ):
            _mark_ez(t)
            continue



        if (
            _canon(_tok(tokens, i)) == "داخل"
            and _tok(tokens, i+1) == "یک"
            and _lemma(tokens, i+2) == "قایق"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حدودی"
            and _tok(tokens, i-1) == "تا"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1) == "خانم"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شخصی"
            and _tok(tokens, i+1) == "کتاب"
            and _tok(tokens, i+2) == "می‌خواند"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مجسمه"
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i+2) == "می‌کند"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ساحل"
            and _tok(tokens, i+1) == "مجسمه"
            and _tok(tokens, i-1) == "شن‌های"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ساحل"
            and _tok(tokens, i+1) == "اصابت"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i+2) == "کرده"

        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "بین"
            and _tok(tokens, i+1) == "روز"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اتاق"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i-2) == "از"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "وارد"
            and _lemma(tokens, i+3) == "شدن"

        ):
            _mark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1) == "نردبان"
            and _tok(tokens, i+2) == "هستند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سمت"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "گربه"
        ):
            _unmark_ez(t)
            continue
        
        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1) == "درخت"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "شاخه"
            and _tok(tokens, i+2) == "آن"
            and _tok(tokens, i+3) == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "توانستن"
            and _tok(tokens, i+1) == "توپ"
            and _tok(tokens, i-1) == "شاید"
            and _tok(tokens, i+2) == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حضور"
            and _tok(tokens, i+1) == "شما"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i-2) == "عرض"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "عکس"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i-1) == "این"
            and _lemma(tokens, i+2) == "خانه"
            and _tok(tokens, i+3) == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i-1) == "فضای"
            and _tok(tokens, i+2) == "عکس"
            and _tok(tokens, i+3) == "یک"
            and _lemma(tokens, i+4) == "خانه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1) == "حمله"
            and _tok(tokens, i-1) == "اینجا"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1) == "خلاص"
            and _tok(tokens, i-1) == "چنگ"
            and _tok(tokens, i+2) == "کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "برداشت"
            and _tok(tokens, i+1) == "من"
            and _tok(tokens, i+2) == "از"
            and _tok(tokens, i+3) == "این"
            and _tok(tokens, i+4) == "تصویر"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "همه"
            and _tok(tokens, i+1) == "ما"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حس"
            and _tok(tokens, i+1) == "بسیار"
            and _tok(tokens, i+1) == "جالبی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نظر"
            and _tok(tokens, i+1) == "من"
            and _tok(tokens, i-1) in {"به","از"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بچه"
            and _tok(tokens, i+1) == "او"
            and _tok(tokens, i+2) == "مشغول"
            and _tok(tokens, i+3) == "بازی"
            and _tok(tokens, i+4) == "توپ"
            and _tok(tokens, i+5) == "است"
        ):
            _mark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i-1) == "تو"
            and _tok(tokens, i+1) == "او"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بازرگان‌های"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "بازارچه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دوره"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "سفره"
            and _tok(tokens, i+3) == "نیستند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آرایش"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "افرادی"
            and _tok(tokens, i+3) == "که"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سمت"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+1) == "یک"
            and _lemma(tokens, i+2) == "گربه"
            and _lemma(tokens, i+3) == "در"
            and _lemma(tokens, i+3) == "حال"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روز"
            and _tok(tokens, i+1) == "مناسبی"
            and _tok(tokens, i+2) == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کار"
            and _lemma(tokens, i+1) == "جدی"
            and _tok(tokens, i+2) == "مشغول"
            and _tok(tokens, i+3) == "باشند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "ساحل"
            and _tok(tokens, i+2) == "نشسته"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "پایش"
            and _tok(tokens, i+2) == "چهارپایه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "تلویزیون"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پرچم"
            and _tok(tokens, i+1) == "سفید"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"حجره‌های","حجرههای"}
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "هستند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پاک"
            and _tok(tokens, i+1) == "خشک"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بشقاب"
            and _tok(tokens, i+1) == "پاک"
            and _tok(tokens, i+2) == "خشک"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانواده"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i-2) == "گمونم"
            and _tok(tokens, i+1) == "روستایی"
            and _tok(tokens, i+2) == "باشند"
        ):
            _mark_ez(t)
            continue
        if (
            _canon(_tok(tokens, i)) == "درختی"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "سرشان"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کتابی"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "این"
            and _tok(tokens, i+3) == "آقا"
        ):            
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "لیوانی"
            and _tok(tokens, i+1) == "جلویشان"
            and _tok(tokens, i+2) == "هست"
        ):            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سیگار"
            and _tok(tokens, i+1) == "دستشان"
            and _tok(tokens, i+2) == "است"
        ):            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "اتاق"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+2) == "وارد"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1) == "مشغول"
            and _tok(tokens, i+2) == "شستشوی"
            and _tok(tokens, i+3) == "ظرف"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1) == "صرفه‌جویی"
            and _tok(tokens, i+2) == "نمی‌کنند"
            and _tok(tokens, i-1) == "را"
            and _tok(tokens, i-2) == "آب"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1) == "سیگار"
            and _tok(tokens, i+2) == "می‌کشد"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1) == "عینک"
            and _tok(tokens, i+2) == "دارد"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرف","آنطرف"}
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "جعبه"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرف","آنطرف"}
            and _tok(tokens, i+1) == "تلویزیون"
            and _tok(tokens, i+2) == "هست"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پسر"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "چهارپایه"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i+1) == "پشت"
            and _tok(tokens, i+2) == "سرشان"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i-1) == "تصویر"
            and _lemma(tokens, i+2) == "سر"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نردیک"
            and _tok(tokens, i-1) in {"عید","اید"}
            and _lemma(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نوشیدنی"
            and _tok(tokens, i+1) == "چیزی"
            and _lemma(tokens, i+2) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"میوه‌هایی","میوههایی"}
            and _tok(tokens, i+1) == "مثل"

        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1) == "دریا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرشان"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "میز"
            and _tok(tokens, i+3) == "نشسته"
        ):
            _mark_ez(t)
            continue

        if _canon(_lemma(tokens, i)) == "بادکنک" and _tok(tokens, i+1) == "هوا" and _tok(tokens, i+1) == "می‌کند":
            _unmark_ez(t)
            continue

        if _canon(_lemma(tokens, i)) in {"روبهروی","روبروی","روبه‌روی"} and _tok(tokens, i+1) == "خانم":
            _mark_ez(t)
            continue

        if _canon(_lemma(tokens, i)) in "جلوی" and _tok(tokens, i+1) == "این":
            _mark_ez(t)
            continue

        if _canon(_lemma(tokens, i)) == "ساعت" and _tok(tokens, i+1) == "دیواری":
            _mark_ez(t)
            continue

        if _canon(_lemma(tokens, i)) == "جای" and _tok(tokens, i+1) == "من" and _tok(tokens, i+2) == "من" and _tok(tokens, i+3) == "این" and _tok(tokens, i+4) == "را" and _tok(tokens, i+5) == "می‌بینم":
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "معمول"
            and _tok(tokens, i-1) == "طبق"
            and _tok(tokens, i+1) == "کنترل"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌طرف","اینطرف"}
            and _tok(tokens, i+1) == "فرار"
            and _tok(tokens, i+1) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "انسان"
            and _tok(tokens, i+1) == "آدم"
            and _tok(tokens, i-1) == "عکس"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پیرمرد"
            and _tok(tokens, i+1) == "موهایش"
            and _tok(tokens, i-1) == "سفید"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "کابینت"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "او"
            and _tok(tokens, i+2) == "است"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آب"
            and _tok(tokens, i+2) == "است"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "آن"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آب"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "میزشان"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "دریا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "چهارپایه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرشان"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "سینک"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "دهانش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "زمین"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "سفره"
            and _tok(tokens, i+2) in {"هفت‌سین","هفتسین"}
            and _tok(tokens, i+3) == "هم"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "آقای"
            and _tok(tokens, i+2) == "دومی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پرچم"
            and _tok(tokens, i+1) == "باد"
            and _tok(tokens, i+2) == "می‌خورد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کتابی"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "تو"
            and _tok(tokens, i+3) == "دستش"
            and _tok(tokens, i+4) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دقت"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i+1) == "مطالعه"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آشپزخانه"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+1) == "ظرف"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کتاب"
            and _tok(tokens, i+1) == "مدل"
            and _tok(tokens, i+2) == "نیست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "جاده"
            and _tok(tokens, i+2) == "که"
            and _tok(tokens, i+3) == "دارد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ظرف‌ها","ظرفها"}
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "نمی‌دانم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1) == "دست"
            and _tok(tokens, i+2) == "او"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i+1) == "او"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دکه‌های"
            and _tok(tokens, i+1) == "دور"
            and _tok(tokens, i+2) == "آن"
            and _tok(tokens, i+3) == "محلی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1) == "قسمت"
            and _tok(tokens, i+1) in {"مغازه‌ها","مغازهها"}
            
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "احیانن"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پارو"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جا"
            and _tok(tokens, i+1) == "عوض"
            and _tok(tokens, i+2) == "کرد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "همه"
            and _tok(tokens, i+1) == "جا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جا"
            and _tok(tokens, i-1) == "همه"
            and _tok(tokens, i+1) == "میوه‌هایی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌طرف","اینطرف"}
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "آقایی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "کار"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌طرفی","اینطرفی"}
            and _tok(tokens, i+1) == "خانم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"همین‌جوری","همینجوری"}
            and _tok(tokens, i+1) == "دیگر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "و"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خلاصه"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "کسی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اتاق"
            and _tok(tokens, i+1) == "نگهبانی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقایان"
            and _tok(tokens, i+1) == "چهار"
            and _tok(tokens, i+2) == "نفر"
            and _tok(tokens, i+3) == "هستند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "پهن"
            and _tok(tokens, i+2) == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "دیوان"
            and _tok(tokens, i+2) == "حافظ"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i+1) == "خاطرم"
            and _tok(tokens, i+2) == "بیشتر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1) == "کوزه"
            and _tok(tokens, i+2) == "دستش"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قشنگی"
            and _tok(tokens, i+1) == "انداختند"
            and _tok(tokens, i-1) == "سفره"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کوزه"
            and _tok(tokens, i-1) == "خانم"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "آن"
            and _tok(tokens, i+3) == "آن"
            and _tok(tokens, i+4) == "آقا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "دختر"
        ):
            _mark_ez(t)
            continue
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پسر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i-1) == "آقا"
            and _lemma(tokens, i+2) == "چهارپایه"
            and _tok(tokens, i-2) == "این"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درواقع"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+1) == "توپ"
            and _tok(tokens, i+3) == "دارند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "گربه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i+1) == "یک‌دانه"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i-2) == "هم"
            and _tok(tokens, i-3) == "پشتش"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "خود"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خود"
            and _tok(tokens, i+1) == "مادر"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "مادرش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+1) == "گربه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "این‌ها"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سیگار"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قلعه"
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "توپ"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"اینجاها","اینجا‌ها"}
            and _tok(tokens, i+1) == "پنجره"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آقاها","آقا‌ها"}
            and _tok(tokens, i+1) == "نردبان"
            and _tok(tokens, i+2) == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i-1) == "تا"
            and _tok(tokens, i-2) == "دور"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "خانم"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرف","آنطرف"}
            and _tok(tokens, i+1) == "نیمکت‌ها"
            and _tok(tokens, i+2) == "چوبی"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i-2) == "دور"
            and _tok(tokens, i-3) == "تا"
            and _tok(tokens, i-4) == "دور"
            and _tok(tokens, i+1) == "کابینت"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i+1) == "دست"
            and _tok(tokens, i+2) == "او"
            and _tok(tokens, i+3) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "میز"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) in {"این","همین"}
            and _tok(tokens, i+2) == "میز"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "طرف"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i+1) == "همین"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بغل"
            and _tok(tokens, i+1) == "همین"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بله"
            and _tok(tokens, i+1) == "دیگر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "و"
            and _tok(tokens, i+1) == "این"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "برج"
            and _tok(tokens, i+2) == "و"
            and _tok(tokens, i+3) == "بارو"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+1) == "کتاب"
        ):
            _mark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "به"
            and _tok(tokens, i+2) == "راحتی"
            and _tok(tokens, i+3) == "نشسته"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حالا"
            and _tok(tokens, i+1) == "وسیله‌ای"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "آنجا"
            and _tok(tokens, i+4) == "هست"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قایق"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i-1) == "تو"
            and _tok(tokens, i+2) == "آقایی"
            and _tok(tokens, i+3) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1) == "قایق"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "آقایی"
            and _tok(tokens, i+4) == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ظرفشویی","ظرف‌شویی"}
            and _tok(tokens, i-1) == "سینک"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "رفته"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "یارو"
            and _tok(tokens, i+1) == "پیراهن"
            and _tok(tokens, i+2) == "سفید"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "رفقا"
            and _tok(tokens, i+1) == "جمع"
            and _tok(tokens, i+2) == "شدند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پسر"
            and _tok(tokens, i+1) == "خردسالی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جا"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+2) == "آمدند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "و"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "طرز"
            and _tok(tokens, i+1) == "گرفتنش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+1) == "شاخه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نزدیک"
            and _tok(tokens, i-1) == "دریاچه‌ای"
            and _tok(tokens, i+1) == "خانه"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پدر"
            and _tok(tokens, i+1) == "کتاب"
            and _tok(tokens, i+2) == "می‌خواند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مادر"
            and _tok(tokens, i+1) == "مشغول"
            and _tok(tokens, i+2) == "انجام"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "خونه‌های"
            and _tok(tokens, i+1) == "ساحلی"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "روی"
            and _tok(tokens, i+1) == "زیلویی"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "معلوم"
            and _tok(tokens, i-1) == "قرار"
            and _tok(tokens, i-2) == "از"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "این"
            and _tok(tokens, i+1) == "بار"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "روی"
            and _tok(tokens, i+1) == "شاخه"
            and _tok(tokens, i+2) == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "بالای"
            and _tok(tokens, i+1) == "درخت"
            and _tok(tokens, i-1) == "می‌افتد"
            and _tok(tokens, i-2) == "توپشان"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "گلدون"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "سرشان"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "پسر"
            and _tok(tokens, i+1) == "بادبادک"
            and _tok(tokens, i+2) == "هوا"
            and _tok(tokens, i+3) == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "تبلتی"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i-1) == "شاید"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "تصویر"
            and _tok(tokens, i+1) == "زیر"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "روز"
            and _tok(tokens, i+1) == "آفتابی"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) in {"پنجرههای","پنجره‌های"}
            and _tok(tokens, i+1) == "قدیمی"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) in {"خونه‌های","خونههای","خانه‌های","خانههای"}
            and _tok(tokens, i+1) == "قدیمی"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) in {"فرشفروش","فرش‌فروش"}
            and _tok(tokens, i+1) == "دوره‌گرد"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) in {"بچهای","بچه‌ای"}
            and _tok(tokens, i+1) in {"توپبازی","توپ‌بازی"}
            and _tok(tokens, i+2) == "می‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) in {"شاهنامهای","شاهنامه‌ای"}
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) in {"پایینیها","پایینی‌ها"}
            and _tok(tokens, i+1) == "پرتاب"
            and _tok(tokens, i-1) == "برای"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        # --- add more of
        # --- add more of your ultra‑specific, surgical Ezafe rules below ---
        # if _canon(_tok(tokens,i)) == "خانم" and _tok(tokens,i+1) == "دیگری": _mark_ez(t); continue
        # if _canon(_tok(tokens,i)) == "سینک" and _canon(_tok(tokens,i+1)) == "آشپزخانه": _mark_ez(t); continue

    return tokens

def _set_pos(tok: dict, pos: str) -> None:
    tok["pos"] = pos

def _apply_manual_pos_overrides(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens):
        # EXAMPLE 2 (yours): same pattern, but force POS
        #   if _canon(tok_i) == "دستی" and tokens[i+1].get("tok") == "بافتنی" and tokens[i+2].get("tok") == "باشد": POS == "NOUN"
        if (
            _canon(_tok(tokens, i)) == "دستی"
            and _tok(tokens, i+1) == "بافتنی"
            and _tok(tokens, i+2) == "باشد"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "باید"
            and _tok(tokens, i+1) in {"بگوییم","باشد"}
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "من"
            and _tok(tokens, i+2) == "باید"
            and _tok(tokens, i+3) == "بگوییم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "باید"
            and _tok(tokens, i+2) == "بگویم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "می‌بینم"
            and _tok(tokens, i-1) == "دیگر"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "باید"
            and _tok(tokens, i+3) == "ببینم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "مغازه‌ها"
            and _tok(tokens, i+3) == "هم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "مبلشون"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "که"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "باز"
            and _tok(tokens, i+3) == "یک"
            and _tok(tokens, i+4) == "خانواده"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "باید"
            and _tok(tokens, i+2) == "ببینم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چه"
            and _tok(tokens, i+2) == "می‌بینم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "بالای"
            and _tok(tokens, i+2) == "مغازه‌ها"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "باز"
            and _tok(tokens, i+2) == "رودخانه"
            and _tok(tokens, i+3) == "هست"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "پرت"
            and _tok(tokens, i+1) == "شده"
            and _tok(tokens, i-1) == "حواسش"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "دو"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i-1) == "هر"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i-1) == "برود"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "بازار"
            and _tok(tokens, i+3) == "هم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "هستم"
            and _tok(tokens, i-1) == "متاسف"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "یکی"
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+2) == "؟"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "گرفتنش"
            and _tok(tokens, i-1) == "طرز"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "هرچی"
            and _tok(tokens, i+1) == "که"
            and _tok(tokens, i+2) == "می‌بینم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "مخده"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i-2) == "از"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "جوون‌تر"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "عشقی"
            and _tok(tokens, i+1) == "باشد"
            and _tok(tokens, i-1) == "کنم"
            and _tok(tokens, i-2) == "فکر"
        ):
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i-1) == "می‌آید"
            and _tok(tokens, i-2) == "دارد"
            and _tok(tokens, i-3) == "هم"
            and _tok(tokens, i-4) == "پسر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "می‌آید"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کی"
            and _tok(tokens, i-1) == "می‌کند"
            and _tok(tokens, i-2) == "نگاه"
            and _tok(tokens, i-3) == "دارد"
            and _tok(tokens, i+1) == "می‌خواهد"
            and _tok(tokens, i+2) == "ببیند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "بعد"
            and _tok(tokens, i+1) == "آقایی"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i-1) == "می‌آید"
            and _tok(tokens, i-2) == "در"
            and _tok(tokens, i-3) == "از"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "پدر"
            and _tok(tokens, i+1) == "قلاب"
            and _tok(tokens, i+2) == "می‌گیرد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "هستم"
            and _tok(tokens, i-1) == "بیرون"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i-1) == "است"
            and _tok(tokens, i-2) == "چوگان"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i+2) == "دست"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "می‌آید"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1) == "دست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "قابلمه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"شیراز","اصفهان","یزد","کرمان","تهران"}
            and _tok(tokens, i-1) == "بازار"
        ):
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i-1) == "تصویر"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "آرام"
            and _tok(tokens, i+1) == "است"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "مسگری"
            and _tok(tokens, i-1) == "مغازه"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "فروشی"
            and _tok(tokens, i-1) == "سفال"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دوتا"
            and _tok(tokens, i+1) == "تابلو"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i-1) == "آقا"
            and _tok(tokens, i+1) == "بالا"
            and _tok(tokens, i+2) == "نشسته"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بنابر"
            and _tok(tokens, i+1) == "دشمنی"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i+1) == "سیر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) in {"عکس","پایین","هنوز","جالب","یک","چیز","یک‌دانه","خانم"}
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "حالا"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i+2) == "دوست"
            and _tok(tokens, i+3) == "دارند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "ها"
            and _tok(tokens, i+3) == "آن‌طوری"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "خیلی"
            and _tok(tokens, i+2) == "زیبا"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i+2) == "دیگر"
            and _tok(tokens, i+3) == "این"
            and _tok(tokens, i+4) == "کار"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "را"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "دوست"
            and _tok(tokens, i+2) == "دارند"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "نقاشی"
            and _tok(tokens, i+1) == "آبرنگ"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "آبرنگ"
            and _tok(tokens, i-1) == "نقاشی"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "می‌گویند"
            and _tok(tokens, i-1) == "یا"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _tok(tokens, i) == "به‌هرحال"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i-1) == "گرفتنش"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "تو"
            and _tok(tokens, i+2) == "دستش"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخر"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "ته"
            and _tok(tokens, i+3) == "بازار"
            and _tok(tokens, i+4) == "است"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i-2) == "ایستاده"
            and _tok(tokens, i+1) == "."
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "ظرف"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "خشک"
            and _tok(tokens, i+4) == "می‌کند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "خلاصه"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "کسی"
            and _tok(tokens, i+3) == "می‌رود"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیرش"
            and _tok(tokens, i+1) == "چه"
            and _tok(tokens, i+2) == "هست"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "پاک"
            and _tok(tokens, i+1) == "کج"
            and _tok(tokens, i-1) == "بشقاب"
            and _tok(tokens, i+2) == "می‌شود"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "پاک"
            and _tok(tokens, i+1) == "خشک"
            and _tok(tokens, i+2) == "می‌کند"
            and _tok(tokens, i-2) == "بشقاب"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کسی"
            and _tok(tokens, i+1) == "می‌رود"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "من"
            and _tok(tokens, i+3) == "به"
            and _tok(tokens, i+4) == "نظرم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "نه"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i+1) == "تا"
            and _tok(tokens, i+2) == "دورش"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالایش"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "بالای"
            and _tok(tokens, i+3) == "سرشان"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "آن"
            and _tok(tokens, i+3) == "آن"
            and _tok(tokens, i+4) == "آقا"
        ):
            _set_pos(t, "ADP")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "آن"
            and _tok(tokens, i+3) == "آقا"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "درواقع"
            and _tok(tokens, i+1) == "دکوراسیون"
            and _tok(tokens, i+2) == "قدیمی"
            and _tok(tokens, i+3) == "خانواده"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخرش"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "تصویر"
            and _tok(tokens, i+3) == "اول"
            and _tok(tokens, i+4) == "که"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالاییه"
            and _tok(tokens, i+1) == "که"
            and _tok(tokens, i-1) == "سگ"
            and _tok(tokens, i-2) == "همان"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "ورودی"
            and _tok(tokens, i-1) == "یک"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "حالا"
            and _tok(tokens, i+1) == "وسیله‌ای"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "آنجا"
            and _tok(tokens, i+4) == "هست"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "دختر"
            and _tok(tokens, i+3) == "هم"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "چوبی"
            and _tok(tokens, i-1) == "اولین"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "کار"
        ):
            _set_pos(t, "DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخری"
            and _tok(tokens, i-1) == "آقای"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "شیشمی"
            and _tok(tokens, i-1) == "هم"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"هیچ‌کدوم","هیچکدوم"}
            and _tok(tokens, i+1) == "هیچ"
            and _tok(tokens, i+1) == "چیزی"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "بچه"
            and _tok(tokens, i-1) == "؟"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i-1) == "؟"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "چشم"
            and _tok(tokens, i-1) == "می‌آید"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدش"
            and _tok(tokens, i-1) == "."
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "شلوار"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "دارد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "کردند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "کنند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "هست"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i-2) == "."
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "تنگ"
            and _tok(tokens, i+1) == "دست"
            and _tok(tokens, i+2) == "او"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "خوب"
            and _tok(tokens, i+2) == "و"
            and _tok(tokens, i+3) == "بدش"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "من"
            and _tok(tokens, i+2) == "چه"
            and _tok(tokens, i+3) == "باید"
            and _tok(tokens, i+3) == "بگوییم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "تنگ"
            and _tok(tokens, i-1) == "دیگر"
            and _tok(tokens, i+1) == "و"
            and _tok(tokens, i+1) == "این‌ها"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبه‌روش","روبهروش"}
            and _tok(tokens, i-1) == "آقایی"
            and _tok(tokens, i-2) == "یک"
        ):
            _set_pos(t, "ADP")
            _set_lemma(t, "روبه‌رو")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بود"
            and _tok(tokens, i-1) == "آن"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "برای"
            and _tok(tokens, i+1) == "هست"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "کرده"
        ):
            _set_pos(t, "PRON")
            continue

        if _canon(_tok(tokens, i)) == "چه" and _tok(tokens, i-1) == "،" and _tok(tokens, i+1) == "کار":
            _set_pos(t, "DET")

        elif _canon(_tok(tokens, i)) == "چه" and _tok(tokens, i-1) == "،":
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "نوشیدنی"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "،"
            and _tok(tokens, i-2) == "هست"
            and _tok(tokens, i-3) == "موز"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "فرشی"
            and _tok(tokens, i-1) == "،"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "؟"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "این‌طرف"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "،"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1) == "،"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1) == "."
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1) == "مثلا"
            and _tok(tokens, i+1) == "کنارش"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنارش"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-2) == "خانواده"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنارش"
            and _tok(tokens, i+1) == "آقایی"
            and _tok(tokens, i+2) == "که"
            and _tok(tokens, i+3) == "نشسته"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "رویش"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "گلدان"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "رویش"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "چایی"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنارش"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "سگ"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "تزیین"
            and _tok(tokens, i+2) == "طلا"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1) == "قایق"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "دیوارش"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "ساعت"
            and _tok(tokens, i+4) == "هست"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنارش"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "گویا"
            and _tok(tokens, i-1) == "بعد"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "تعقیبش"
            and _tok(tokens, i-1) == "کرد"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-2) == "خانواده"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"خنک‌کننده","خنککننده"}
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "یا"
            and _tok(tokens, i-2) == "می‌بینم"
            and _tok(tokens, i-3) == "اشتباه"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "یکی"
            and _tok(tokens, i+2) == "از"
            and _tok(tokens, i+3) == "پسرها"
            and _tok(tokens, i+3) == "قلاب"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "قلاب"
            and _tok(tokens, i+1) == "می‌گیرد"
        ):
            _unmark_ez
            continue


        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "امم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "فکر"
            and _tok(tokens, i+3) == "کنم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آمدند"
            and _tok(tokens, i+1) == "دارند"
            and _tok(tokens, i+2) == "فوتبال"
            and _tok(tokens, i+3) == "یا"
            and _tok(tokens, i+4) == "والیبال"
            and _tok(tokens, i+5) == "بازی"
            and _tok(tokens, i+5) == "می‌کنند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دوتا"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) in {"پرنده","کبوتر"}
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"
            and _tok(tokens, i+1) == "طبیعتا"
            and _tok(tokens, i+2) == "براق"
            and _tok(tokens, i+3) == "می‌شود"
        ):
            _set_pos(t, "NOUN")
            continue
        if (
            _canon(_tok(tokens, i)) in {"آقاسگه","سگه","سگ"}
            and _tok(tokens, i+1) == "دنبال"
            and _tok(tokens, i+2) == "گربه"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "منزل"
            and _tok(tokens, i+1) == "آشپزخانه"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالایش"
            and _tok(tokens, i-1) == "رفته"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+3) == "واژگون"
            and _tok(tokens, i+4) == "می‌شود"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "زدن"
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i-1) == "اود"
            and _tok(tokens, i-2) == "ادای"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "رقصیدن"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i-1) == "یا"
            and _tok(tokens, i-2) == "زدن"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "دیگر"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "حرکت"
            and _tok(tokens, i+2) == "آه"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1) == "درون"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "می‌آیند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "آقا"
            and _tok(tokens, i+3) == "نشسته"
            and _tok(tokens, i-1) == "."
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "یک"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "آقا"
            and _tok(tokens, i+2) == "نشسته"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "نمی‌دانم"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "نمی‌دانم"
            and _tok(tokens, i+1) == "کار"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _set_pos(t, "DET")

        elif _canon(_tok(tokens, i)) == "چه" and _tok(tokens, i-1) == "نمی‌دانم" and _tok(tokens, i+1) == "کار" and _tok(tokens, i+2) == "دارد" and _tok(tokens, i+3) == "می‌کند":
            _set_pos(t, "DET")

        elif _canon(_tok(tokens, i)) == "چه" and _tok(tokens, i-1) == "نمی‌دانم":
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "شیرینی"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کوزه"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "بعد"
            and _tok(tokens, i+1) == "شیرینی"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "یکی"
            and _tok(tokens, i+1) == "می‌آید"
            and _tok(tokens, i+2) == "کمک"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "وق"
        ):
            _set_lemma(t,"واق")
            continue

        if (
            _canon(_tok(tokens, i)) == "احیانن"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "زده"
            and _tok(tokens, i+2) == "شده"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "رفت"
            and _tok(tokens, i+1) == "و"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+2) == "آمد"
            and _tok(tokens, i+3) == "هستند"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبالش"
            and _tok(tokens, i-1) == "می‌دود"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "آمد"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i-2) == "رفت"
            and _tok(tokens, i-3) == "در"
            and _tok(tokens, i+1) == "هستند"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "رفتن"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "درخت"
            and _tok(tokens, i-1) == "حال"
            and _tok(tokens, i-2) == "در"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبالش"
            and _tok(tokens, i-1) == "می‌دود"
            and _tok(tokens, i-2) == "سگ"
            and _tok(tokens, i+1) == "که"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "طبق"
            and _tok(tokens, i+1) == "معمول"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلویش"
            and _tok(tokens, i+1) == "ایستاده"
        ):
            _set_pos(t, "ADP")
            continue

        if _canon(_lemma(tokens, i)) in {"پیکنیکی","پیک‌نیکی"} and _tok(tokens, i-1) == "حالت":
            _set_pos(t, "ADJ")
            continue

        if _canon(_lemma(tokens, i)) == "دیگر" and _tok(tokens, i-1) == "." and _tok(tokens, i+1) == "شمع‌های" and _tok(tokens, i+2) == "قشنگ":
            _set_pos(t, "ADV")
            continue

        if _canon(_lemma(tokens, i)) == "دنبالش" and _tok(tokens, i-1) == "می‌دود":
            _set_pos(t, "ADP")
            continue

        if _canon(_lemma(tokens, i)) == "سیزده‌به‌در":
            _set_pos(t, "PROPN")
            continue

        if _canon(_lemma(tokens, i)) in {"سیزده‌به‌در","سیزدهبهدر","سیزده‌بهدر","سیزدهبه‌در"}:
            _set_pos(t, "PROPN")
            continue

        if _canon(_lemma(tokens, i)) == "چهارتا":
            _set_pos(t, "NUM")
            continue

        if _canon(_lemma(tokens, i)) == "بعد" and _tok(tokens, i+1) == "با" and _tok(tokens, i+2) == "پارو" and _tok(tokens, i+3) == "دارد":
            _set_pos(t, "ADV")
            continue

        if _canon(_lemma(tokens, i)) == "بعد" and _tok(tokens, i+1) == "این" and _tok(tokens, i+2) == "طرف" and _tok(tokens, i+3) == "هم":
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "افتادن"
            and _tok(tokens, i+1) == "از"
            and _tok(tokens, i+2) == "قایق"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1) == "."
            and _tok(tokens, i+1) == "توپشان"
            and _tok(tokens, i+2) == "رفته"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+1) == "هست"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "هست"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بود"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بگوییم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "هر"
            and _tok(tokens, i+1) == "می‌بینم"
            and _tok(tokens, i+2) == "بگوییم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بود"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "را"
            and _tok(tokens, i-2) == "این"
            and _tok(tokens, i+1) == "بگوییم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i+1) == "زیاد"
            and _tok(tokens, i+2) == "باز"
            and _tok(tokens, i+3) == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیاد"
            and _tok(tokens, i-1) == "آب"
            and _tok(tokens, i+1) == "باز"
            and _tok(tokens, i+2) == "کرده"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1) == "روی"
            and _tok(tokens, i+1) == "درخت"
        ):
            _set_pos(t,"DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1) == "کوزه"
            and _tok(tokens, i+2) == "آب"
            and _tok(tokens, i+3) == "دستش"
            and _tok(tokens, i+4) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نظر"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "عکسی"
            and _tok(tokens, i+3) == "چیزی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خودکاری"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "مدادی"
            and _tok(tokens, i+2) == "چیزی"
            and _tok(tokens, i+3) == "که"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"عکس‌هایی","عکسهایی"}
            and _tok(tokens, i-1) == "قاب"

        ):
            _set_pos(t,"NOUN")
            _set_lemma(t,"عکس")
            t["had_plural_suffix"] = True
            continue

        if (
            _canon(_tok(tokens, i)) in {"گربه‌شان","گربهشان"}
        ):
            _set_lemma(t,"گربه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"نوشابه‌اش","نوشابهاش"}
        ):
            _set_lemma(t,"نوشابه")
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i-1) == "کوزه"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "هوا"
            and _tok(tokens, i-1) == "بادبادک"
            and _tok(tokens, i+1) == "کردن"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"بیلچه‌ی","بیلچهی"}
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "آقاپسر"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اضافه"
            and _tok(tokens, i-1) == "آنجا"
            and _tok(tokens, i-2) == "هم"
            and _tok(tokens, i-3) == "چیزی"
            and _tok(tokens, i+1) == "دارم"
            and _tok(tokens, i+2) == "می‌بینم"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "اضافه"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i+1) == "اینکه"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "اضافه"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i+1) == "یک"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کردند"
            and _tok(tokens, i-1) == "خشک"
            and _tok(tokens, i-2) == "مشغول"
            and _tok(tokens, i+1) == "ظرف‌ها"
            and _tok(tokens, i+2) == "می‌باشد"
        ):
            _set_token(t,"کردن")
            _set_lemma(t,"کردن")
            t["morph_segments"] = []
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خواندند"
            and _tok(tokens, i-1) == "کتاب"
            and _tok(tokens, i-2) == "حال"
            and _tok(tokens, i-3) == "در"
        ):
            _set_token(t,"خواندن")
            _set_lemma(t,"خواندن")
            t["morph_segments"] = []
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "است"
            and _tok(tokens, i+1) == "هست"
            and _tok(tokens, i-2) == "گربه"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "جالباسی"
            and _tok(tokens, i-1) == "رگال"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "راست"
            and _tok(tokens, i-1) == "سمت"
            and _tok(tokens, i-2) == "از"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "درخت"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "حیوان"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1) == "عکسش"
            and _tok(tokens, i+2) == "را"
        ):
            _set_pos(t, "SCONJ")
            continue



        if _canon(_lemma(tokens, i)) in {"روبهروی","روبروی"} and _tok(tokens, i+1) == "است":
            _set_pos(t, "ADP")
            continue

        if _canon(_lemma(tokens, i)) in {"سنبالا","سن‌بالا"}:
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "خب"
            and _tok(tokens, i+1) == "مسگری"
            and _tok(tokens, i+2) == "می‌گویم"
            and _tok(tokens, i+3) == "هست"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "خونه‌های"
            and _tok(tokens, i+2) == "حالت"
            and _tok(tokens, i+3) == "قدیمی"
            and _tok(tokens, i+4) == "هم"
            and _tok(tokens, i+5) == "هست"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "سفره"
            and _tok(tokens, i+2) == "هفت‌سین"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "همین‌ها"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADV")
            continue


        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "شیر"
            and _tok(tokens, i+2) == "آب"
            and _tok(tokens, i+3) == "هم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "چهارپایه"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سر"
            and _tok(tokens, i+2) == "این"
            and _tok(tokens, i+3) == "آقا"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i-1) == "تو"
            and _tok(tokens, i+1) == "گذاشته"
            and _tok(tokens, i+2) == "شده"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "افتادن"
            and _tok(tokens, i+1) == "از"
            and _tok(tokens, i-1) == "حال"
            and _tok(tokens, i+2) == "قایق"
            and _tok(tokens, i-2) == "در"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "باشند"
            and _tok(tokens, i-1) == "روستایی"
            and _tok(tokens, i-2) == "خانواده"
            and _tok(tokens, i-3) == "یک"
            and _tok(tokens, i-4) == "گمونم"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "باشم"
            and _tok(tokens, i-1) == "دیده"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "باشم"
            and _tok(tokens, i-1) == "کرده"
            and _tok(tokens, i-2) == "عرض"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "بله"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "من"
            and _tok(tokens, i+3) == "فکر"
            and _tok(tokens, i+4) == "کنم"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بله"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "این"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلویشان"
            and _tok(tokens, i+1) == "هست"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کامواش"
            and _tok(tokens, i-1) == "سبد"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بله"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "این"
            and _tok(tokens, i+3) == "یک"

        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بله"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "سنبل"
            and _tok(tokens, i+3) == "هست"

        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "داشته"
            and _tok(tokens, i+1) == "باشیم"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "باشیم"
            and _tok(tokens, i-1) == "داشته"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "آشپزخانه"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "آقا"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "کتاب"
            and _tok(tokens, i-1) == "از"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "گربه"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سمت"
            and _tok(tokens, i+2) == "چپ"
            and _tok(tokens, i+3) == "ساعت"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _tok(tokens, i) == "دونه‌دونه"
            and _tok(tokens, i+1) == "اینها"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "بگوییم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _tok(tokens, i) == "دوتا"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "کبوتر"
            and _tok(tokens, i+3) == "می‌بینم"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _tok(tokens, i) == "دارد"
            and _tok(tokens, i+1) == "با"
            and _tok(tokens, i+2) == "بیلچه"
            and _tok(tokens, i-1) == "نشسته"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "مشغول"
            and _tok(tokens, i+2) == "بردن"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i+1) == "گذاشته"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "سگ"
            and _tok(tokens, i+3) == "شروع"
            and _tok(tokens, i+4) == "می‌کند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1) == "مشغول"
            and _tok(tokens, i+2) == "نوشیدنی"
            and _tok(tokens, i+3) == "یا"
            and _tok(tokens, i+4) == "صبحانه"
            and _tok(tokens, i+5) == "هستند"
        ):
            _set_pos(t, "SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "چند"
            and _tok(tokens, i+3) == "دستگاه"
            and _tok(tokens, i+4) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1) == "اگر"
            and _tok(tokens, i+2) == "اشتباه"
            and _tok(tokens, i+3) == "نکنم"
            and _tok(tokens, i+4) == "،"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1) == "همه"
            and _tok(tokens, i+2) == "ما"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "دختر"
            and _tok(tokens, i+2) == "دیگر"
            and _tok(tokens, i+3) == "هم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "برداشت"
            and _tok(tokens, i+1) == "من"
            and _tok(tokens, i+2) == "از"
            and _tok(tokens, i+3) == "این"
            and _tok(tokens, i+4) == "تصویر"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "باشند"
            and _tok(tokens, i-1) == "مشغول"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "سمت"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i-1) == "به"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دستی"
            and _tok(tokens, i+1) == "بافتنی"
            and _tok(tokens, i+2) == "باشد"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "گربه"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "شاخه"
            and _tok(tokens, i-1) == "از"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "شاخه"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "الاچیقی"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "وارد"
            and _tok(tokens, i+1) == "می‌شود"

        ):
            _set_pos(t, "ADJ")
            continue
        if (
            _canon(_tok(tokens, i)) == "آماده"
            and _tok(tokens, i+1) == "می‌شود"

        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "اماده"
            and _tok(tokens, i+1) == "می‌شود"

        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "آماده"
            and _tok(tokens, i+1) == "می‌شود"

        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "وارد"
            and _tok(tokens, i+1) == "شده"

        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بافتن"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "کاموا"

        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "تلویزیون"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "روبروی"
            and _tok(tokens, i+1) == "خانم"
            and _tok(tokens, i+2) == "و"
            and _tok(tokens, i+3) == "آقا"
        ):
            _set_pos(t, "ADP")
            continue


        if (
            _canon(_tok(tokens, i)) == "نظیر"
            and _tok(tokens, i+1) == "این‌ها"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "نشیمن"
            and _tok(tokens, i-1) == "اتاق"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرشان"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "ظرف"
            and _tok(tokens, i+2) == "کلوچه"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) in {"اینجوری","این‌جوری"}
            and _tok(tokens, i+1) == "یا"
            and _tok(tokens, i+2) == "چادر"
            and _tok(tokens, i+3) == "است"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "توانستن"
            and _tok(tokens, i+1) == "توپ"
            and _tok(tokens, i-1) == "شاید"
            and _tok(tokens, i+2) == "را"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "جالب"
            and _tok(tokens, i-1) == "حضورتان"
            and _tok(tokens, i-2) == "به"
            and _tok(tokens, i-3) == "عرض"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بله"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i-2) == "هست"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "تلویزیون"
            and _tok(tokens, i+2) == "خوابیده"

        ):
            _set_pos(t, "ADP")
            continue


        if (
            _canon(_tok(tokens, i)) == "یکیشون"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "می‌رود"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "احتمالا"
            and _tok(tokens, i+2) == "یا"
            and _tok(tokens, i+3) == "خریده"
            and _tok(tokens, i+4) == "یا"
            and _tok(tokens, i+5) == "دارد"
            and _tok(tokens, i+6) == "برای"
            and _tok(tokens, i+7) == "عرضه"
            and _tok(tokens, i+8) == "می‌برد"

        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "قرآن"
            and _tok(tokens, i+2) == "می‌خواند"

        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "شستن"
            and _tok(tokens, i-1) == "برای"
            and _tok(tokens, i-2) == "است"
            and _tok(tokens, i-3) == "نبسته"
            and _tok(tokens, i-4) == "آب"

        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1) == "روی"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخر"
            and _tok(tokens, i-1) == "تا"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "اتاق"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+2) == "وارد"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "اول"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i-2) == "دوباره"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i-1) == "هستند"
            and _tok(tokens, i+1) == "هستند"
            and _tok(tokens, i-2) == "سگ"
            
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "گرفتن"
            and _tok(tokens, i-1) == "آماده"
            and _tok(tokens, i+1) == "است"
            
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1) == "عکسش"
            and _tok(tokens, i-1) == "را"
            and _tok(tokens, i+2) == "دیدیم"

        ):
            _set_pos(t, "SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "تا"
            and _tok(tokens, i+1) == "کرده"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "باشد"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "اهرم"
            and _tok(tokens, i+1) == "دارد"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_lemma(tokens, i)) == "کارایی"
            and _tok(tokens, i+1) == "دارد"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) in {"جلوتر","جلو‌تر"}
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "قدم"
            and _tok(tokens, i+2) == "می‌زند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "کتاب"
            and _tok(tokens, i+2) == "می‌خواند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "کتاب"
            and _tok(tokens, i+2) == "می‌خواند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "چپقش"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "می‌کشد"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "با"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i+2) == "ساعتش"
            and _tok(tokens, i-2) == "فردی"
        ):
            _set_pos(t, "AUX")
            continue


        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آب"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i-2) == "آب"
            and _tok(tokens, i-3) == "و"
            and _tok(tokens, i-4) == "آسمان"
        ):
            _set_pos(t, "NOUN")
            continue



        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بگوییم"
            and _tok(tokens, i+2) == "من"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بگوییم"
            and _tok(tokens, i-1) == "نمی‌دانم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "هست"
            and _tok(tokens, i-1) == "این"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "هست"
            and _tok(tokens, i-1) == "حالا"
            and _tok(tokens, i-2) == "هم"
            and _tok(tokens, i-3) == "این"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "تنگ"
            and _tok(tokens, i+1) == "آب"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i+1) == "همه"
            and _tok(tokens, i+2) == "ریشه"
            and _tok(tokens, i+3) == "دارد"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i-1) == "آقای"
            and _tok(tokens, i-2) == "یک"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "همینطور"
            and _tok(tokens, i-1) == "هم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1) == "آه"
            and _tok(tokens, i+2) == "،"
            and _tok(tokens, i+3) == "آب"
            and _tok(tokens, i-1) == "نفر"
            and _tok(tokens, i-2) == "یک"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "ایستادن"
            and _tok(tokens, i-1) == "حال"
            and _tok(tokens, i-2) == "در"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "کار"
        ):
            _set_pos(t, "DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "چیز"
            and _tok(tokens, i+3) == "دیگر"
            and _tok(tokens, i+4) == "نمی‌دانم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "دیگر"
            and _tok(tokens, i+3) == "نمی‌دانم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "نمی‌دانم"
            and _tok(tokens, i-1) == "ظرف‌ها"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"یه‌دونه","یهدونه"}
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "که"
            and _tok(tokens, i+3) == "همین"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) in {"تهران","قزوین","کرمان","اصفهان","یزد"}

        ):
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالایش"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "پایینش"
            and _tok(tokens, i+3) == "همه"
            and _tok(tokens, i+4) == "ریشه"
            and _tok(tokens, i+5) == "دارد"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "ساعت"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == {"همین‌جوری","همینجوری"}
            and _tok(tokens, i+1) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "همین"
            and _tok(tokens, i+2) == "."
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"همینطرف","همین‌طرف"}
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i-1) == "باز"
            and _tok(tokens, i+2) == "طرف"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آهان"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "وسطیه"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "این"
            and _tok(tokens, i+3) == "انار"
            and _tok(tokens, i+4) == "است"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبالش"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "به"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "گل"
            and _tok(tokens, i+2) == "نرگس"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "پای"
            and _tok(tokens, i+2) == "این"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبالش"
            and _tok(tokens, i-1) == "می‌رود"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "می‌گوید"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "؟"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "به"
            and _tok(tokens, i+2) == "راحتی"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "بازار"
            and _tok(tokens, i+3) == "شیراز"
            and _tok(tokens, i+4) == "است"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "فکر"
            and _tok(tokens, i+3) == "کنم"
            and _tok(tokens, i+4) == "عشقی"
            and _tok(tokens, i+5) == "باشد"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "یکی"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "نردبان"
            and _tok(tokens, i+3) == "را"
            and _tok(tokens, i+4) == "نگه"
            and _tok(tokens, i+5) == "داشته"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) in {"دونهدونه","دونه‌دونه"}
            and _tok(tokens, i-1) == "حالا"
            and _tok(tokens, i+1) == "اینها"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "را"
            and _tok(tokens, i+4) == "بگوییم"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "مریم"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "موقع"
            and _tok(tokens, i+3) == "که"

        ):
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "حافظ"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "احتمالا"
        ):
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "مریم"
            and _tok(tokens, i+1) == "نگاه"
            and _tok(tokens, i+2) == "کن"

        ):
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "مریم"
            and _tok(tokens, i-1) == "ببین"

        ):
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "سمت"
            and _tok(tokens, i-1) == "تقریبا"
            and _tok(tokens, i-2) == "آمده"
            and _tok(tokens, i+1) == "این"

        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "مبل"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "است"
            and _tok(tokens, i-2) == "نردبان"
            and _tok(tokens, i+1) == "شد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بوده"
            and _tok(tokens, i+2) == "این"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "ظرف‌های"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"بشقاب‌های","بشقابهای"}
            and _tok(tokens, i+1) == "مختلفی"
            and _tok(tokens, i+2) == "هست"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آینه"
            and _tok(tokens, i+1) == "داخلش"
            and _tok(tokens, i+2) == "است"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زمانی"
            and _tok(tokens, i-1) == "آقای"

        ):
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "ها"
            and _tok(tokens, i-1) == "روی"
            and _tok(tokens, i+1) == "آن"

        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"همه‌چی","همهچی"}
            and _tok(tokens, i-1) == "آماده"

        ):
            _unmark_ez
            continue

        



        # --- add more of your POS fixes below ---
        # if _canon(_tok(tokens,i)) == "دنبال" and _pos(tokens,i+1).startswith("NOUN"): _set_pos(t,"ADP"); continue
        # if _canon(_tok(tokens,i)) == "وارد" and _pos(tokens,i+1).startswith(("NOUN","PROPN")): _set_pos(t,"ADJ"); continue

    return tokens

def _set_lemma(tok: dict, lemma: str) -> None:
    tok["lemma"] = lemma

def _apply_manual_lemma_overrides(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    for i, t in enumerate(tokens):
        if _canon(_lemma(tokens, i)) in {"روبهروی","روبروی"} and _tok(tokens, i+1) == "است":
            _set_lemma(t, "روبه‌رو")

        
        if _canon(_lemma(tokens, i)) in {"سفره","سفر‌ه"}:
            _set_lemma(t, "سفره")
            continue

        if _canon(_lemma(tokens, i)) == "آبی" and _tok(tokens, i+1) == "که" and _tok(tokens, i+2) == "در" and _tok(tokens, i+3) == "حال" and _tok(tokens, i+4) == "ریزش" and _tok(tokens, i+5) == "است" :
            _set_lemma(t, "آب")
            continue

        if _canon(_tok(tokens, i)) == "ایستان" and _tok(tokens, i-1) == "حال" and _tok(tokens, i-2) == "در":
            _set_lemma(t, "ایستادن")
            continue

        if _canon(_tok(tokens, i)) == "مدلش":
            _set_lemma(t, "مدل")
            continue

    return tokens

def _set_token(tok: dict, token: str) -> None:
    tok["tok"] = token

def _apply_token_lemma_overrides(tokens: list[dict]) -> list[dict]:
    n = len(tokens)
    to_delete = set()
    for i, t in enumerate(tokens):
        if _canon(_tok(tokens, i)) == "بدهد" and _tok(tokens, i-1) == "بدهد":
            _set_token(t, "بده")
            continue

    

    for i, t in enumerate(tokens):
        if _canon(_tok(tokens, i)) == "بدهد" and _tok(tokens, i+1) in {"بده","بدهد"}:
            _set_token(t, "بده")
            continue

    for i, t in enumerate(tokens):
        if _canon(_tok(tokens, i)) == "گرفتند" and _tok(tokens, i-1) == "حالت" and _tok(tokens, i+1) == "فیلم" and _tok(tokens, i+2) == "است":
            _set_token(t, "گرفتن")
            continue

    for i, t in enumerate(tokens):
        if _canon(_tok(tokens, i)) == "داداشه"and _tok(tokens, i+1) == ".":
            _set_token(t,"داداش")
            _set_lemma (t, "داداش")
            continue

    for i, t in enumerate(tokens):
        if _canon(_tok(tokens, i)) == "ایستان" and _tok(tokens, i-1) == "حال" and _tok(tokens, i-2) == "در":
            _set_token(t, "ایستادن")
            continue

        if _canon(_tok(tokens, i)) == "قفس" and _tok(tokens, i+1) == "است" and _tok(tokens, i+2) == "کتاب":
            _set_token(t, "قفسه")
            _set_lemma(t, "قفسه")
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "چی"
            and _tok(tokens, i+1) == "ست"
        ):
            _set_token(t, "چیست")
            continue


        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i-1) == "است"
            and _tok(tokens, i-2) == "چوگان"
            and _tok(tokens, i+1) == "چیست"
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if (
            _canon(_tok(tokens, i)) == "ست"
            and _tok(tokens, i-1) == "چیست"
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if _canon(_tok(tokens, i)) == "زیبای" and _tok(tokens, i+1) == "ی":
            _set_token(t, "زیبایی")
            _set_lemma(t, "زیبا")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ی"
            and _tok(tokens, i-1) == "زیبایی"
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if _canon(_tok(tokens, i)) == "زیبایی" and _tok(tokens, i+1) == "طاق‌های":
            _set_token(t, "زیبایی")
            _set_lemma(t, "زیبا")
            _mark_ez(t)
            continue

        if _canon(_tok(tokens, i)) == "آن" and _tok(tokens, i+1) == "دور" and _tok(tokens, i-1) == "از" and _tok(tokens, i+2) == "گربه" and _tok(tokens, i+3) == "را":
            _set_pos(t, "DET")
            _unmark_ez(t)
            continue

        if _canon(_tok(tokens, i)) == "فرشی" and _tok(tokens, i+1) == "پهن" and _tok(tokens, i+2) == "کردند" :
            _unmark_ez(t)
            continue

        if _canon(_tok(tokens, i)) == "کوتاه" and _tok(tokens, i+1) == "شدن" and _tok(tokens, i-1) == "آستین" and _tok(tokens, i-2) == "پیراهن" and _tok(tokens, i-3) == "با":
            _unmark_ez(t)
            continue

        if _canon(_tok(tokens, i)) in {"زیاده","زیاد"} and _tok(tokens, i+1) == "ه":
            _set_token(t, "زیاده")
            _set_lemma(t, "زیاد")
            continue

        if _canon(_tok(tokens, i)) == "قفس" and _tok(tokens, i-1) == "آن" and _tok(tokens, i-2) == "روی" and _tok(tokens, i-3) == "گل":
            _set_token(t, "قفسه")
            _set_lemma(t, "قفسه")
            continue

        if _canon(_tok(tokens, i)) == {"جه‌اند","جهاند"} and _tok(tokens, i-1) == "اتحاد":
            _set_token(t, "جهان")
            _set_lemma(t, "جهان")
            _set_pos(t, "NOUN")
            continue


        if _canon(_tok(tokens, i)) in {"میزنتش","می‌زنتش"}:
            _set_lemma(t, "زدن")
            continue

        if _canon(_tok(tokens, i)) in {"آه‌اند","آهاند"}:
            _set_token(t, "آهان")
            _set_lemma(t, "آهان")
            _set_pos(t, "INTJ")
            continue

        if _canon(_tok(tokens, i)) in {" و ","و"}:
            _unmark_ez(t)
            continue

        if _canon(_tok(tokens, i)) in {" ! ","!"}:
            _unmark_ez(t)
            _set_pos(t, "PUNCT")
            continue

        if _canon(_tok(tokens, i)) == "در" and _tok(tokens, i+1) in {"وارد","داخل"} and _tok(tokens, i-1) == "از" and _tok(tokens, i+2) == "می‌شود":
            _unmark_ez(t)
            _set_pos(t, "NOUN")
            continue

        if _canon(_tok(tokens, i)) == "خونهی":
            _set_token(t, "خونه‌ی")
            _set_lemma(t, "خونه")
            continue

        if _tok(tokens, i) in {"پایه‌ه‌اش","پایههاش"}:
            _set_token(t, "پایه‌هایش")
            _set_lemma(t, "پایه")
            continue

        if _tok(tokens, i) in {"ریش‌ه‌اش","ریشهاش"}:
            _set_token(t, "ریش‌هایش")
            _set_lemma(t, "ریش")
            continue



        # فقط "است" وسط را حذف کن (قفسه [است] کتاب است)
        if (
            _canon(_tok(tokens, i)) == "است"
            and _tok(tokens, i-1) == "قفسه"
            and _tok(tokens, i+1) == "کتاب"
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if (
            _canon(_tok(tokens, i)) == "است"
            and _tok(tokens, i-1) == "گربه"
            and _tok(tokens, i+1) == "می‌پرد"
            and _tok(tokens, i+2) == "می‌رود"
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue



        if (
            _canon(_tok(tokens, i)) == "ه"
            and _tok(tokens, i-1) in {"زیاده","زیاد"}
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if (
            _canon(_tok(tokens, i)) == "ساعت"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "سرش"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "غذا"
            and _tok(tokens, i+1) == "آماده"
            and _tok(tokens, i+2) == "می‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "محلی"
            and _tok(tokens, i-1) == "لباس"
            and _tok(tokens, i+1) == "تنش"
            and _tok(tokens, i+2) == "است"
            
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مادر"
            and _tok(tokens, i+1) == "دور"
            and _tok(tokens, i+2) == "میز"
            and _tok(tokens, i+3) == "نشسته‌اند"
            
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "دنبال"
            and _tok(tokens, i+4) == "می‌کند"
            
        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "دنبال"
            and _tok(tokens, i+4) == "می‌کند"
            
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"گربهها","گربه‌ها"}
            and _tok(tokens, i+1) == "احساس"
            and _tok(tokens, i+2) == "امنیت"
            and _tok(tokens, i+3) == "نمی‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "می‌کند"
            and _tok(tokens, i-1) == "را"
            and _tok(tokens, i-2) == "گربه"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کی"
            and _tok(tokens, i+1) == "هستند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "هستش"
            and _tok(tokens, i-1) == "بازار"
            and _tok(tokens, i-2) == "یک"
            and _tok(tokens, i-3) == "مثل"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "تا"
            and _tok(tokens, i+1) == "می‌خواهد"
            and _tok(tokens, i+2) == "برود"
            and _tok(tokens, i+3) == "بالا"
        ):
            _set_pos(t, "SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "عقب"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "می‌افتد"
            and _tok(tokens, i-1) == "سمت"
            and _tok(tokens, i-2) == "به"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کیوسکی"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "قرار"
            and _tok(tokens, i-1) == "مثل"
            and _tok(tokens, i-2) == "گرفته"
        ):
            _set_pos(t, "ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "هستیم"
            and _tok(tokens, i-1) == "طبیعت"
            and _tok(tokens, i-2) == "از"
            and _tok(tokens, i-3) == "بخشی"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "جای"
            and _tok(tokens, i+1) == "خیلی"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "زیبایی"
            and _tok(tokens, i+1) == "دیده"
            and _tok(tokens, i-1) == "سبز"
            and _tok(tokens, i-2) == "رنگ"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پلی"
            and _tok(tokens, i+1) == "آنجا"
            and _tok(tokens, i+2) == "هست"
        ):
            _set_pos(t, "NOUN")
            _unmark_ez(t)
            continue

    

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i+2) == "نمی‌فهمم"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _tok(tokens, i) == "آن‌طرف‌تر"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "آقایی"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _tok(tokens, i) == "این‌طرف‌تر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "قایق"
            and _tok(tokens, i+3) == "پارویی"
            and _tok(tokens, i+4) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "چیزی"
            and _tok(tokens, i+1) == "ذهنم"
            and _tok(tokens, i+2) == "نمی‌رسد"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "بالایش"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i-1) == "رفته"
            and _tok(tokens, i+2) == "واژگون"
            and _tok(tokens, i+3) == "می‌شود"
        ):
            _unmark_ez(t)
            _set_pos(t, "NOUN")
            continue

        if (
            _tok(tokens, i) == "دارند"
            and _tok(tokens, i+1) == "در"
            and _tok(tokens, i+2) == "حال"
            and _tok(tokens, i+2) == "توپ‌بازی"
            and _tok(tokens, i+3) == "هستند"
        ):
            _unmark_ez(t)
            _set_pos(t, "AUX")
            continue

        if (
            _tok(tokens, i) == "همین‌طور"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "سری"
            and _tok(tokens, i+2) == "افراد"
            and _tok(tokens, i+3) == "دیگر"
        ):
            _unmark_ez(t)
            _set_pos(t, "ADV")
            continue

        if (
            _tok(tokens, i) == "بالای"
            and _tok(tokens, i+1) == "اسکله"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "بالای"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "اسکله"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "بالای"
            and _tok(tokens, i+1) == "سرش"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "زندان"
            and _tok(tokens, i+1) == "فرار"
            and _tok(tokens, i+2) == "کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تابلویی"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "تلویزیون"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1) == "گربه‌ای"
            and _tok(tokens, i+2) == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "وسط"
            and _tok(tokens, i+1) == "می‌بینیم"
            and _tok(tokens, i-1) == "تصویر"
            and _tok(tokens, i-2) == "در"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "رد"
            and _tok(tokens, i+1) == "نمی‌کند"
            and _tok(tokens, i-1) == "را"
            and _tok(tokens, i-2) == "آب"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "آن‌ها"
            and _tok(tokens, i-1) == "بعد"
            and _tok(tokens, i+2) == "متوجه"
            and _tok(tokens, i+3) == "نمی‌شوند"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "زیبا"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"آقاگربه‌ای","آقاگربهای"}
            and _tok(tokens, i+1) == "نشسته"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدی"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i-1) == "تصویر"
            and _tok(tokens, i+2) == "قایق"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقاسگه"
            and _tok(tokens, i+1) == "دنبال"
            and _tok(tokens, i+2) == "گربه"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقاسگه"
            and _tok(tokens, i+1) == "دنبال"
            and _tok(tokens, i+2) == "گربه"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدی"
            and _tok(tokens, i-1) == "تصویر"
            and _tok(tokens, i+1) == "،"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "عده‌ای"
            and _tok(tokens, i+1) == "چهار"
            and _tok(tokens, i+2) == "نفر"
            and _tok(tokens, i+3) == "هستند"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "بعدی"
            and _tok(tokens, i-1) == "تصویر"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "بعدی"
            and _tok(tokens, i-1) == "تصویر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "قایق"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "است"
            and _tok(tokens, i-1) == "قفسه"
            and _tok(tokens, i+1) == "کتاب"
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if (
            _canon(_tok(tokens, i)) == "است"
            and _tok(tokens, i-1) == "است"
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if (
            _canon(_tok(tokens, i)) == "است"
            and _tok(tokens, i-1) == "سگ"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "می‌آید"
            and _tok(tokens, i+3) == "به"
            and _tok(tokens, i+4) == "طرف"
            and _tok(tokens, i+5) == "دریا"
        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if (
            _canon(_tok(tokens, i)) == "است"
            and _tok(tokens, i-1) == "سگ"
            and _tok(tokens, i-2) == "بعد"
            and _tok(tokens, i+1) == "می‌خواهد"
            and _tok(tokens, i+2) == "برود"

        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if (
            _canon(_tok(tokens, i)) == "است"
            and _tok(tokens, i-1) == "دختر"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "می‌خواهد"
            and _tok(tokens, i+3) == "بگیرد"

        ):
            to_delete.add(i)  # حذف کامل خود توکن
            continue

        if (
            _canon(_tok(tokens, i)) in {"دونهدونه","دونه‌دونه"}
            and _tok(tokens, i-1) == "حالا"
            and _tok(tokens, i+1) == "اینها"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "را"
            and _tok(tokens, i+4) == "بگوییم"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بدو"
            and _tok(tokens, i+1) == "آفرینش"

        ):
            _set_pos(t, "NOUN")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نه"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "آقایی"
            and _tok(tokens, i-3) == "یک"

        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "هیچکدوم"
            and _tok(tokens, i+1) == "سواد"
            and _tok(tokens, i+2) == "ندارند"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چپ"
            and _tok(tokens, i+1) == "و"
            and _tok(tokens, i+2) == "راست"
            and _tok(tokens, i+3) == "ندارد"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "راست"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i-2) == "چپ"
            and _tok(tokens, i+1) == "ندارد"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "تعقیبش"
            and _tok(tokens, i+1) == "کرد"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "واژگون"
            and _tok(tokens, i+1) == "شدن"
            and _tok(tokens, i+2) == "است"
            and _tok(tokens, i-1) == "حال"
            and _tok(tokens, i-2) == "در"
        ):
            _set_pos(t, "ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شدن"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "واژگون"
            and _tok(tokens, i-2) == "حال"
            and _tok(tokens, i-3) == "در"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "بازار"
            and _tok(tokens, i+1) == "سنتی"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue
        if (
            _canon(_tok(tokens, i)) == "الاغ"
            and _tok(tokens, i-1) == "با"
            and _tok(tokens, i+1) == "حرکت"
            and _tok(tokens, i+2) == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "گاری"
            and _tok(tokens, i-1) == "با"
            and _tok(tokens, i+1) == "حرکت"
            and _tok(tokens, i+2) == "دادند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درواقع"
        ):
            _unmark_ez(t)
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چیز"
            and _tok(tokens, i-1) == "اتحاد"
            and _tok(tokens, i+1) == "خوب"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تصمیمات"
            and _tok(tokens, i+1) == "مهمی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نتیجه"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "واحد"
            and _tok(tokens, i+2) == "می‌شود"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اسکله"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "فلزی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قایق"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "پارویی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پدر"
            and _tok(tokens, i+1) == "سیگار"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"لباس‌های","لباسهای"}
            and _tok(tokens, i+1) == "سنتی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"دهنده‌ی","دهندهی"}
            and _tok(tokens, i-1) == "نشان"
            and _pos(tokens, i+1) != "PUNCT"
        ):
            _mark_ez(t)
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _mark_ez(t)
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پای"
            and _tok(tokens, i+1) == "درخت"
        ):
            _mark_ez(t)
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "خانه‌اشان"
        ):
            _mark_ez(t)
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیرش"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "درمی‌رود"
        ):
            _unmark_ez(t)
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "درواقع"
        ):
            
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"سهتا","سه‌تا"}
            and _tok(tokens, i-1) == "دو"
            and _tok(tokens, i-1) == "هم"

        ):
            
            _set_pos(t,"NUM")
            _unmark_ez(t)
            continue
        
        if (
            _canon(_tok(tokens, i)) == "درواقع"
            and _tok(tokens, i+1) == "تعقیب"
            and _tok(tokens, i+2) == "و"
            and _tok(tokens, i+3) == "گریز"
            and _tok(tokens, i+4) == "است"
        ):
            
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "تو"
            and _tok(tokens, i+2) == "بغلش"
            and _tok(tokens, i+3) == "است"
        ):
            
            _set_pos(t,"PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "احتمالا"
            and _tok(tokens, i+2) == "فکر"
            and _tok(tokens, i+3) == "می‌کنم"
        ):
            
            _set_pos(t,"PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "برای"
        ):
            
            _set_pos(t,"PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن‌ها"
        ):
            
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+1) == "دور"
        ):
            
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "مانند"
            and _tok(tokens, i-1) == "خانه"
            and _tok(tokens, i+1) == "است"
        ):
            
            _set_pos(t,"ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مانندی"
            and _tok(tokens, i-1) == "کلاسور"
            and _tok(tokens, i+1) == "چیزی"
        ):
            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "معروف"
            and _tok(tokens, i-1) == "قول"
            and _tok(tokens, i+1) == "ریش"
        ):
            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیرانداز"
            and _tok(tokens, i+1) == "انداختن"
        ):
            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i-1) == "نشسته"
            and _tok(tokens, i+2) == "معذرت"
            and _tok(tokens, i+3) == "می‌خواهم"
            and _tok(tokens, i+4) == "،"
            and _tok(tokens, i+5) == "الاغ"
        ):
            _set_pos(t, "ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+2) == "مردم"
            and _tok(tokens, i+3) == "می‌آیند"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "حمل"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+2) == "می‌کنند"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرفش","آنطرفش"}
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i-1) == "بعد"
            and _tok(tokens, i+2) == "مثل"
            and _tok(tokens, i+3) == "یک"
            and _tok(tokens, i+4) == "رودخانه"
        ):
            _set_pos(t, "NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرفش","آنطرفش"}
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "گربه"
            and _tok(tokens, i+4) == "مانندی"
        ):
            _set_pos(t, "NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرفش","آنطرفش"}
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "که"
            and _tok(tokens, i+3) == "باز"
            and _tok(tokens, i+4) == "چندین"
            and _tok(tokens, i+5) == "نفر"
            and _tok(tokens, i+6) == "هستند"
        ):
            _set_pos(t, "NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "گذاشته"
            and _tok(tokens, i+2) == "روی"
            and _tok(tokens, i+3) == "پرواز"
            and _tok(tokens, i+4) == "کرده"
        ):
            _set_pos(t, "SCONJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"سه‌چرخ","سهچرخ"}
            and _tok(tokens, i-1) == "چهارچرخ"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i+1) == "پرده‌ای"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "مانند"
            and _tok(tokens, i-1) == "بیل"
            and _tok(tokens, i+1) == "است"
        ):
            
            _set_pos(t,"ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دارند"
            and _tok(tokens, i+1) == "فوتبال"
            and _tok(tokens, i+2) == "،"
            and _tok(tokens, i+3) == "چیز"
            and _tok(tokens, i+4) == "،"
            and _tok(tokens, i+5) == "والیبال"
            and _tok(tokens, i+6) == "بازی"
            
        ): 
            _set_pos(t,"AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "خانومی"
            and _tok(tokens, i+1) == "بشقاب"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1) == "برای"
            and _tok(tokens, i-2) == "شده"
            and _tok(tokens, i-3) == "کج"
        ):
            
            _set_pos(t,"PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آخر"
            and _tok(tokens, i+1) == "فوتبال"
            and _tok(tokens, i-1) == "تصویر"
            and _tok(tokens, i+2) == "بازی"
            and _tok(tokens, i+3) == "می‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مانند"
            and _tok(tokens, i-1) == "بشقاب"
        ):
            
            _set_pos(t,"ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بسی"
        ):
            
            _set_pos(t,"PART")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "فکر"
            and _tok(tokens, i+1) == "می‌کنند"
        ):
            
            _set_pos(t,"PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تلویزیون"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "ورزشکار"
            and _tok(tokens, i+3) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بسی"
            and _tok(tokens, i-1) == "چه"
        ):
            
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نردبان"
            and _tok(tokens, i-1) == "جای"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "چیزی"
        ):
            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مانند"
            and _tok(tokens, i-1) == "سگ"
            and _tok(tokens, i+1) == "است"
        ):
            
            _set_pos(t,"ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "غذا"
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مردم"
            and _tok(tokens, i+1) == "خرید"
            and _tok(tokens, i+2) == "فروش"
            and _tok(tokens, i+3) == "می‌کنند"
        ):
            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "توپ"
            and _tok(tokens, i+1) == "دستشان"
            and _tok(tokens, i+2) == "است"
        ):
            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"خانهی","خانه‌ی"}
            and _tok(tokens, i+1) == "آن‌ها"
        ):
            
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جوان"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i-2) == "بعد"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "بچه"
            and _tok(tokens, i+3) == "را"
        ):
            
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "است"
        ):
            
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "گذاشته"
        ):
            
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "آب"
            and _tok(tokens, i+2) == "چه"
            and _tok(tokens, i+3) == "باشد"
        ):
            
            _set_pos(t,"CCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایه"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i-1) == "ولی"
        ):
            
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ظروف‌های","ظروفهای"}
            and _tok(tokens, i+1) == "متعددی"
        ):
            
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایه"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i-1) == "ولی"
        ):
            
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کوچک"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "سگ"
            and _tok(tokens, i+3) == "مانند"
            
        ):
            
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "باشد"
            and _tok(tokens, i-1) == "آب"
            and _tok(tokens, i-2) == "چه"
        ):
            
            _set_pos(t,"CCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "باشد"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "این"
        ):
            
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) in {"عدهای","عده‌ای"}
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "وسایل"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "خانم"
            and _tok(tokens, i+3) == "هم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فیدل"
            and _tok(tokens, i+1) == "کاسترو"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پرچمی"
            and _tok(tokens, i+1) == "آویزان"
            and _tok(tokens, i+2) == "به"
            and _tok(tokens, i+3) == "آن"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سبد"
            and _tok(tokens, i+1) in {"همراه‌شون","همراهشون"}
            and _tok(tokens, i+2) == "هست"
            
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بچه"
            and _tok(tokens, i+1) == "او"
            and _tok(tokens, i+2) == "دارد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i+1) == "نشان‌"
            and _tok(tokens, i+2) == "دهنده"
        ):
            _unmark_ez(t)
            continue
        if (
            _canon(_tok(tokens, i)) == "دهنده"
            and _tok(tokens, i-1) == "نشان‌"
            and _tok(tokens, i+1) == "این"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کوچکی"
            and _tok(tokens, i-1) == "دفترچه"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درحقیقت"
            and _tok(tokens, i+1) == "دخترخانم"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درحقیقت"
            and _tok(tokens, i+1) == "فلزی"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درحقیقت"
            and _tok(tokens, i+1) == "شمعدان"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i-1) == "است"
            and _tok(tokens, i-2) == "برده"
            and _tok(tokens, i-3) == "را"
            and _tok(tokens, i-4) == "توپ"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1) == "اینجا"
            and _tok(tokens, i+2) == "هست"
        ):
            _set_pos(t,"ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "عقب"
            and _tok(tokens, i-1) == "آن"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i-1) == "برود"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "نردبان"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "ساعت"
            and _tok(tokens, i+2) == "هشت"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i-1) == "شیر"
            and _tok(tokens, i+1) == "باز"
            and _tok(tokens, i+2) == "مانده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مچاله"
            and _tok(tokens, i+1) == "نشسته"
        ):
            _set_pos(t,"ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ظرف"
            and _tok(tokens, i+1) == "خشک"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقای"
            and _tok(tokens, i+1) == "دیگر"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قایقی"
            and _tok(tokens, i+1) == "راه"
            and _tok(tokens, i-1) == "آب"
            and _tok(tokens, i-2) == "درون"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i+1) == "قایقی"
            and _tok(tokens, i-1) == "درون"
            and _tok(tokens, i+2) == "راه"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دریا"
            and _tok(tokens, i-1) == "لب"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جای"
            and _tok(tokens, i+1) == "دیگری"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"
            and _tok(tokens, i-1) == "یا"
            and _tok(tokens, i-2) == "گربه"
        ):
            _set_pos(t, "NOUN")
            _set_lemma(t, "سگ")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "می‌رفت"
        ):
            _set_pos(t, "NOUN")
            _set_lemma(t, "سگ")
            continue

        if (
            _canon(_tok(tokens, i)) == "رفت"
            and _tok(tokens, i-1) == "یک‌دانه"
            and _tok(tokens, i+1) == "هست"
        ):
            _set_pos(t, "NOUN")
            _set_lemma(t, "رف")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "مثل"
            and _tok(tokens, i+2) == "اینکه"
            and _tok(tokens, i+3) == "کتاب"
            and _tok(tokens, i+4) == "می‌خواند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "یکی"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "آنجا"
            and _tok(tokens, i+3) == "ایستاده"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) in {"ماهی‌ای","ماهیای"}
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "می‌گیرد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرفش","آنطرفش"}
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "قاب"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقاپسر"
            and _tok(tokens, i+1) == "انار"
            and _tok(tokens, i+2) == "و"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "وانمود"
            and _tok(tokens, i+1) == "کند"
            and _tok(tokens, i+2) == "که"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1) == "بعد"
            and _tok(tokens, i+1) == "طرف"
        ):
            _set_pos(t, "DET")
            _set_lemma(t,"این")
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i+1) == "میز"
            and _tok(tokens, i+2) == "نشسته‌اند"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "میز"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "تو"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کابینت"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i-1) == "آن"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i+1) == "لب"
            and _tok(tokens, i+2) == "آب"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قطار"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "بالا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1) == "قطار"
            and _tok(tokens, i-2) == "یک"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بچه"
            and _tok(tokens, i+1) == "بادبادک"
            and _tok(tokens, i+2) == "هوا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "انتهای"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "پل"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "موبایل"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "وسط"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) in {"سفره‌ای","سفره‌ی","سفره"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نزدیک"
            and _tok(tokens, i+1) == "ساحل"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "خوابیده"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i-2) == "و"
            and _tok(tokens, i-3) == "می‌برد"
            and _tok(tokens, i-4) == "لذت"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقای"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "چیزی"
        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "خلاصه"
            and _tok(tokens, i+1) == "بازاری"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "در"
            and _tok(tokens, i+4) == "دولنگه"
            and _tok(tokens, i+5) == "است"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبهرو","روبه‌رو"}
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "آن"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "روبه‌رو"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == {"دنباله‌اش","دنتبالهاش"}
        ):
            _set_lemma(t,"دنباله")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "هست"
            and _tok(tokens, i-1) == "این"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+1) == "طرف"
            
        ):
            _set_pos(t,"DET")
            _set_lemma(t,"این")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) in {"دیوار","نوردبان","نردبان"}
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "فکر"
            and _tok(tokens, i+1) == "می‌کنم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "همین"
        ):
            _set_pos(t, "ADV")
            continue


        if (
            _tok(tokens, i) == "می‌گردند"
            and _tok(tokens, i-1) == "چیزی"
            and _tok(tokens, i-2) == "دنبال"
        ):
            _set_lemma(t, "گشتن")
            continue

        if (
            _tok(tokens, i) == "دارد"
            and _tok(tokens, i+1) == "با"
            and _tok(tokens, i+2) == "کمک"
            and _tok(tokens, i+3) == "آن"
            and _tok(tokens, i+4) == "آقاپسر"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _tok(tokens, i) == "جلوی"
            and _tok(tokens, i+1) == "در"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "که"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "در"
            and _tok(tokens, i-1) == "جلوی"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "که"
        ):
            _set_pos(t, "NOUN")
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "روی"
            and _tok(tokens, i+1) == "مبل"
            and _tok(tokens, i+2) == "نشسته"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "دارد"
            and _tok(tokens, i+1) in {"قران","فرآن"}
            and _tok(tokens, i+2) == "،"
            and _tok(tokens, i+3) == "نمی‌دانم"
            and _tok(tokens, i+4) == "قران"
            and _tok(tokens, i+5) == "یا"
            and _tok(tokens, i+6) == "دعا"
            and _tok(tokens, i+7) == "می‌خواند"
        ):
            _set_pos(t, "AUX")
            continue


        if (
            _canon(_tok(tokens, i)) in {"قایقهای","قایق‌های","قایق"}
            and _tok(tokens, i+1) == "کاغذی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"بهبه","والا","آمم","به‌به","اممم","به به","سلام علیکم","سلامعلیکم"}

        ):
            _set_pos(t, "INTJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "والا"

        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "درواقع"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌یکی","اینیکی"}
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "که"
            and _tok(tokens, i+3) == "دارد"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i-2) == "از"
            and _tok(tokens, i+3) == "می‌بیند"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "نردبان"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+2) == "آقایی"
            and _tok(tokens, i-2) == "در"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "باد"
            and _tok(tokens, i+1) == "تکونش"
            and _tok(tokens, i+2) == "می‌دهد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آشپزخانه"
            and _tok(tokens, i+1) == "پرده"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سلام"
            and _tok(tokens, i+1) == "علیکم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i+1) == "همدیگر"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "در"
            and _tok(tokens, i+2) == "هم"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"
            and _tok(tokens, i-1) == "یا"
            and _tok(tokens, i-2) == "گربه"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"

        ):
            _set_pos(t, "NOUN")
            _set_lemma(t, "سگ")
            continue

        if (
            _canon(_tok(tokens, i)) == "نو"
            and _tok(tokens, i+1) == "مبارک"
            and _tok(tokens, i-1) == "سال"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "هرحال"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "نفر"
            and _tok(tokens, i+3) == "نشسته"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آینه"
            and _tok(tokens, i-1) == "بعد"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i+1) == "همدیگر"
            and _tok(tokens, i+2) == "نشسته‌اند"

        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "در"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "می‌گردند"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "کتابخونه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "ظرف‌ها"
            and _tok(tokens, i+3) == "پول"
            and _tok(tokens, i+4) == "ساز"
            and _tok(tokens, i+5) == "است"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"ظرف‌ها","ظرفها"}
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i-2) == "یعنی"
            and _tok(tokens, i+1) == "پول"
            and _tok(tokens, i+2) == "ساز"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"یکدانه","یک‌دانه"}
            and _tok(tokens, i+1) == "آقا"
            and _tok(tokens, i+2) == "هست"
        ):
            _set_pos(t,"NUM")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌طرف","اینطرف"}
            and _tok(tokens, i+1) == "یک‌دانه"
            and _tok(tokens, i+2) == "آقا"
            and _tok(tokens, i+3) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "پیشخوان"
            and _tok(tokens, i+2) == "کابینت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "سه‌پایه"
            and _tok(tokens, i-1) == "رفته"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "تلویزیون"
            and _tok(tokens, i-1) == "گربه"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلویش"
            and _tok(tokens, i+1) == "حالا"
        ):
            _set_pos(t, "ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "یکی"
            and _tok(tokens, i+1) == "اینجا"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "دیگری"
            and _tok(tokens, i+3) == "نمی‌بینم"
        ):
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگ‌های"
            and _tok(tokens, i+1) == "مسی"
        ):
            _set_pos(t, "NOUN")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌چنین","اینچنین"}
            and _tok(tokens, i+1) == "حالتی"
            and _tok(tokens, i-1) == "،"
            and _tok(tokens, i+2) == "."
        ):
            _set_pos(t, "DET")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبالش"
            and _tok(tokens, i+1) == "با"
            and _tok(tokens, i+2) == "لباس"
            and _tok(tokens, i+3) == "محلی"
            and _tok(tokens, i+4) == "و"
            and _tok(tokens, i+5) == "کوزه"
            and _tok(tokens, i+6) == "و"
            and _tok(tokens, i+7) == "این"
            and _tok(tokens, i+8) == "حرف‌ها"
            and _tok(tokens, i+9) == "دارد"
            and _tok(tokens, i+10) == "راه"
            and _tok(tokens, i+11) == "می‌رود"
        ):
            _set_pos(t, "ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درختی"
            and _tok(tokens, i+1) == "فرشی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مختلف"
            and _tok(tokens, i+1) == "درونش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _lemma(tokens, i) == "زیبا"
            and _tok(tokens, i+1) == "دیده"
            and _tok(tokens, i+2) == "می‌شود"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "راست"
            and _tok(tokens, i+1) == "حرکت"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+2) == "می‌کنیم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i+1) == "می‌رویم"
        ):
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "آن"
            and _tok(tokens, i+1) == "دور"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+2) == "گربه"
            and _tok(tokens, i+3) == "را"
        ):
            _set_pos(t, "DET")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i+1) == "درونش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بادبانی"
            and _tok(tokens, i-1) == "قایق"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"سهپایهی","سهپایه","سه‌پایه"}
            and _tok(tokens, i+1) == "زیر"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"اون‌یکی","اونیکی"}
            and _tok(tokens, i+1) == "روی"
            and _tok(tokens, i+2) == "چه"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"اون‌یکی","اونیکی"}
            and _tok(tokens, i+1) == "روی"
            and _tok(tokens, i+2) == "مبل"
            and _tok(tokens, i+3) == "نشسته"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "باز"
            and _tok(tokens, i+2) == "است"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "فروشی"
            and _tok(tokens, i-1) == "مس"
            and _tok(tokens, i+1) == "است"

        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "پلی"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "آنجا"
            and _tok(tokens, i+2) == "هست"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کاموایش"
            and _tok(tokens, i+1) == "هم"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1) == "پنجره"
            and _tok(tokens, i+2) == "پیدا"
            and _tok(tokens, i+3) == "است"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i-2) == "دورنما"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "درخت"
            and _tok(tokens, i+2) == "بیاورد"

        ):
            _set_pos(t, "ADP")
            continue

        
        if (
            _canon(_tok(tokens, i)) in {"اینیکی","این‌یکی"}
            and _tok(tokens, i-1) == "حالا"

        ):
            _set_pos(t, "PRON")
            continue

        

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "فرار"
            and _tok(tokens, i-1) == "."
            and _tok(tokens, i+2) == "کرده"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "باز"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+2) == "شده"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "نیست"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "قشنگ"
            and _tok(tokens, i+1) == "پوشیدن"
            and _tok(tokens, i-1) == "."
            and _tok(tokens, i+2) == "."
        ):
            _unmark_ez(t)
            continue



        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "به"
            and _tok(tokens, i+2) == "ذهنتان"
            and _tok(tokens, i+3) == "می‌آید"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) in {"یک‌دانه","یک"}
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"یک‌دانه","یکدانه"}
            and _tok(tokens, i+1) == "در"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "گربه"
            and _tok(tokens, i+1) == "بالایش"
            and _tok(tokens, i+2) == "رفته"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "کمد"
            and _tok(tokens, i+1) == "است"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i-1) == "می‌آید"
            and _tok(tokens, i-2) == "دارد"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "یک"
            and _tok(tokens, i+4) == "درخت"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i-1) == "اینجا"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "آقاپسر"
            and _tok(tokens, i+4) == "سن"
            and _tok(tokens, i+4) == "دارد"

        ):
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقاپسر"
            and _tok(tokens, i+1) == "سن"
            and _tok(tokens, i+2) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "هرکدومش"
            and _tok(tokens, i+1) == "برای"
            and _tok(tokens, i+2) == "خودشان"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "درخت"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"چیچی","چی‌چی"}
            and _tok(tokens, i+1) == "ریخته"
            and _tok(tokens, i+2) == "افتاده"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1) == "جلوی"
            and _tok(tokens, i+1) == "خانه‌شان"
        ):
            _set_pos(t, "NOUN")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پای"
            and _tok(tokens, i-1) == "زیر"
            and _tok(tokens, i+1) == "او"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پسر"
            and _tok(tokens, i+1) == "من"
            and _tok(tokens, i+2) == "آمده"
            and _tok(tokens, i+3) == "امتحان"
            and _tok(tokens, i+4) == "داده"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i+1) == "ایشان"
            and _tok(tokens, i+2) == "راه"
            and _tok(tokens, i+3) == "می‌رود"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "با"
            and _tok(tokens, i+2) == "بیلچه"
            and _tok(tokens, i+3) == "نمی‌دانم"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌طرف","اینطرف"}
            and _tok(tokens, i+1) == "توپ"
            and _tok(tokens, i-2) == "،"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بیرون"
            and _tok(tokens, i+1) == "است"
        ):
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue



        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "در"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1) == "بالای"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "سگی"
            and _tok(tokens, i+3) == "هست"

        ):
            _set_token(t, "بالا")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+1) == "ما"
            and _tok(tokens, i+2) == "می‌بینیم"
        ):
            _set_token(t, "بالا")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "بالا"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1) == "بالای"
        ):
            _set_pos(t, "ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بیرون"
            and _tok(tokens, i-1) == "یک‌دانه"
            and _tok(tokens, i+1) == "است"
        ):
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "خانه‌شان"
            and _tok(tokens, i+2) == "اینجا"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درخت"
            and _tok(tokens, i-1) == "جلوی"
            and _tok(tokens, i+1) == "پارک"
            and _tok(tokens, i+2) == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ایناها"
            and _tok(tokens, i+1) == "آها"
            and _tok(tokens, i+2) == "میز"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i-1) == "رفته"
            and _tok(tokens, i-2) == "هم"
            and _tok(tokens, i-3) == "خودش"
            
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "هرکدوم"
            and _tok(tokens, i+1) == "یک‌دانه"
            and _tok(tokens, i+2) == "الاغ"
            and _tok(tokens, i+3) == "گرفتند"
            
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "برگی"
            and _tok(tokens, i+1) == "خدا"
            and _tok(tokens, i+2) == "می‌داند"
            and _tok(tokens, i-1) == "طور"
            and _tok(tokens, i-2) == "چه"
            
        ):
            _set_pos(t,"NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i-1) == "می‌پرد"
            and _tok(tokens, i-2) == "بالا"
            and _tok(tokens, i-3) == "از"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدش"
            and _tok(tokens, i+1) == "چه"
            and _tok(tokens, i+2) == "کار"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "راست"
            and _tok(tokens, i-1) == "بیایم"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "موز"
            and _tok(tokens, i-1) == "که"
            and _tok(tokens, i-2) == "هم"
            and _tok(tokens, i-3) == "این"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "انگاری"
            and _tok(tokens, i+1) == "که"
            and _tok(tokens, i-1) == "بشقاب‌ها"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1) == "توله"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1) == "لباس"
            and _tok(tokens, i+2) == "محلی"
            and _tok(tokens, i+3) == "پوشیده"
            
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i-1) == "می‌خواهند"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "بریزند"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) in {"برگهایش","برگ‌هایش"}
            and _tok(tokens, i+1) == "مثل"
            and _tok(tokens, i+2) == "من"
            and _tok(tokens, i+3) == "پاییز"
            and _tok(tokens, i+4) == "کرده"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بافتنی"
            and _tok(tokens, i-1) == "نشست"
            and _tok(tokens, i+1) == "بافت"

        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "نمی‌دانم"
            and _tok(tokens, i+1) == "چه"
            and _tok(tokens, i+2) == "هست"

        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "خاصی"
            and _tok(tokens, i+3) == "نمی‌بینم"

        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "نمی‌دانم"
            and _tok(tokens, i+2) == "چه"
            and _tok(tokens, i+3) == "کار"
            and _tok(tokens, i+4) == "می‌کند"

        ):
            _set_pos(t,"AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبالش"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "خانم"
            and _tok(tokens, i-3) == "یک"
            and _tok(tokens, i+1) == "است"

        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "یک"

        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1) == "تصویر"
            and _tok(tokens, i+1) == "بگذار"
            and _tok(tokens, i+2) == "ببینم"

        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "ها"
            and _tok(tokens, i-1) == "رفت"
            and _tok(tokens, i-2) == "یادم"
        ):
            _set_pos(t,"INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "الاغی"
            and _tok(tokens, i-1) == "حیوان"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-2) == "یک"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "ها"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "آن"
        ):
            _set_pos(t,"INTJ")
            continue


        if (
            _canon(_tok(tokens, i)) in {"میوه‌ی","میوهی"}
            and _tok(tokens, i+1) == "گرد"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مثل"
            and _tok(tokens, i+1) == "من"
            and _tok(tokens, i+2) == "پاییز"
            and _tok(tokens, i+3) == "کرده"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "لیوان"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بچه"
            and _tok(tokens, i+1) == "توپ‌بازی"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_lemma(tokens, i)) == "زیبا"
            and _tok(tokens, i+1) == "طاق‌های"
            and _tok(tokens, i+2) == "بازارهای"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1) == "سعی"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نردبان"
            and _tok(tokens, i+1) == "توپ"
            and _tok(tokens, i+2) == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قدبلند"
            and _tok(tokens, i+1) == "یک‌دانه"
            and _tok(tokens, i+2) == "از"
            and _tok(tokens, i+3) == "این"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "صحیح"
            and _tok(tokens, i+1) == "حاضر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قشنگ"
            and _tok(tokens, i+1) == "اتاق"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تعقیبش"
            and _tok(tokens, i+1) == "برود"
            and _tok(tokens, i-1) == "می‌رود"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "نزدیک"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "باشد"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "سوی"
            and _tok(tokens, i+1) == "آقا"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i+1) == "که"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدش"
            and _tok(tokens, i+1) == "کجا"
            and _tok(tokens, i+2) == "بروم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدش"
            and _tok(tokens, i+1) == "اینجا"
            and _tok(tokens, i+2) == "نردبان"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "ها"
            and _tok(tokens, i-1) in {"می‌کنند","نیست"}
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"طاق‌های","طاقهای"}
            and _tok(tokens, i+1) == "بازارهای"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ناصرالدین"
            and _tok(tokens, i+1) == "شاه"
        ):
            _set_pos(t, "PROPN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اصطلاح"
            and _tok(tokens, i+1) == "تماشاچی"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i+2) == "هستند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i+2) == "هم"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سلامتی"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i+1) == "خوش"
            and _tok(tokens, i+2) == "بگذرد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیز"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "ساحلی"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "یازارهای"
            and _tok(tokens, i+1) == "قدیمی"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دعایی"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "دارد"
            and _tok(tokens, i+3) == "می‌خواند"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چه"
            and _tok(tokens, i+2) == "بگوییم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "نمی‌بینم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "خلاصه"
            and _tok(tokens, i+2) == "همین"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "عزیزم"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کسیش"
            and _tok(tokens, i+1) == "هست"
            and _tok(tokens, i-1) == "یا"
        ):
            _set_pos(t, "PRON")
            continue





        if (
            _canon(_tok(tokens, i)) == "مظفرالدین"
            and _tok(tokens, i+1) == "شاه"
        ):
            _set_pos(t, "PROPN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "هرکدومش"
            and _tok(tokens, i+1) == "برای"
            and _tok(tokens, i+2) == "خودشان"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "به"
            and _tok(tokens, i-1) == "هر"
            and _tok(tokens, i+2) == "ذهنتان"
        ):
            _set_pos(t, "PRON")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1) == "دریا"
            and _tok(tokens, i-1) == "برود"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "‌ظرف‌ها"
            and _tok(tokens, i+3) == "پول‌ساز"
            and _tok(tokens, i+4) == "است"
        ):
            _set_pos(t, "SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "هرکدوم"
            and _tok(tokens, i+1) == "یک‌دانه"
            and _tok(tokens, i+2) == "الاغ"
            and _tok(tokens, i+3) == "گرفتند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1) == "یه‌خورده"
            and _tok(tokens, i+2) == "وقتش"
            and _tok(tokens, i+3) == "گذشته"
        ):
            _set_pos(t, "SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1) == "می‌آید"
            and _tok(tokens, i+2) == "دارد"
            and _tok(tokens, i+3) == "آقاپسری"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "می‌گفتم"
            and _tok(tokens, i+1) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دوتا"
            and _tok(tokens, i+1) == "هستند"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) in {"حاج‌آقا","حاجآقا","حاجاقا"}
            and _tok(tokens, i+1) == "گناه"
            and _tok(tokens, i+2) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"بادبادکها","بادبادک‌ها","هواپیماها"}
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i+2) == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "مثل"
            and _tok(tokens, i+2) == "اینکه"
            and _tok(tokens, i+3) == "آینه"
            and _tok(tokens, i+4) == "را"
            and _tok(tokens, i+5) == "نگاه"
            and _tok(tokens, i+6) == "می‌کند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1) == "بالای"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "سگی"
            and _tok(tokens, i+3) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "طور"
            and _tok(tokens, i-1) == "چه"
            and _tok(tokens, i+1) == "برگی"
            and _tok(tokens, i+2) == "خدا"
            and _tok(tokens, i+3) == "می‌داند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"چیز","جای"}
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "پای"
            and _tok(tokens, i+2) == "او"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"چیزهای","چیز","چراغ","بازارهای"}
            and _tok(tokens, i+1) in {"کوچولو","قدیمی","بلند"}
            and _tok(tokens, i+2) in {"بود","است","هست"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "بقیه‌اش"
            and _tok(tokens, i+2) == "هیچ"
            and _tok(tokens, i+3) == "چیزی"
        ):
            _set_pos(t, "ADV")
            continue
        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "خاصی"
            and _tok(tokens, i+3) == "نیست"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "دیگر"
            and _tok(tokens, i+3) == "نموند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i+1) == "سوی"
            and _tok(tokens, i+2) == "آقا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "عین"
            and _tok(tokens, i+2) == "حال"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "عین"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+2) == "حال"
        ):
            _set_pos(t, "NOUN")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حال"
            and _tok(tokens, i-1) == "عین"
            and _tok(tokens, i-2) == "در"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "معلوم"
            and _tok(tokens, i+2) == "نیست"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چه"
            and _tok(tokens, i-1) == "نمی‌دانم"
            and _tok(tokens, i+2) == "هست"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "بس"
            and _tok(tokens, i+2) == "هست"
            and _tok(tokens, i+3) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "هست"
            and _tok(tokens, i-2) == "بس"
            and _tok(tokens, i-3) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "خلاصه"
            and _tok(tokens, i+1) == "همین"
            and _tok(tokens, i+2) == "است"
            and _tok(tokens, i-1) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "چیزهایی"
            and _tok(tokens, i+3) == "هست"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "ترشی"
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i+2) == "می‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیاد"
            and _tok(tokens, i+1) == "داشت"
            and _tok(tokens, i-1) == "یکی"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبهرو","روبه‌رو"}
            and _tok(tokens, i+1) == "."
            and _tok(tokens, i-1) == "."
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"همه‌چی","همهچی"}
            and _tok(tokens, i+1) == "آماده"
            and _tok(tokens, i-1) == "که"
            and _tok(tokens, i-2) == "هم"
            and _tok(tokens, i-3) == "اینجا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i+1) == "دوباره"
            and _tok(tokens, i+2) == "می‌زند"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i+1) == "طلایی"
            and _tok(tokens, i+2) == "رنگ"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i+1) == "کوچولوتر"
            and _tok(tokens, i-1) == "یک‌دانه"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "هست"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) in {"چیز","آب"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نه"
            and _tok(tokens, i+1) == "اینجا"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "محوطه‌ای"
            and _tok(tokens, i+4) == "هست"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "نه"
            and _tok(tokens, i+1) == "صندلی"
            and _tok(tokens, i+2) == "نه"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "نه"
            and _tok(tokens, i+1) == "نیست"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1) == "پرده‌ای"
            and _tok(tokens, i+1) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چیز"
            and _tok(tokens, i+1) == "خاصی"
            and _tok(tokens, i+2) == "نیست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نه"
            and _tok(tokens, i-1) == "صندلی"
            and _tok(tokens, i-2) == "نه"
        ):
            _set_pos(t, "INTJ")
            continue


        if (
            _canon(_tok(tokens, i)) in {"ظرف‌های","ظرفهای"}
            and _tok(tokens, i+1) == "مسی"
            and _tok(tokens, i+2) in {"بود","است"}
        ):
            _mark_ez(t)
            continue



        if (
            _canon(_tok(tokens, i)) in {"باغچهی","باغچه‌ی"}
            and _tok(tokens, i+1) == "ما"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i+1) == "محسوسی"
            and _tok(tokens, i+2) == "نیست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "لوازم"
            and _tok(tokens, i+1) == "مسی"
            and _tok(tokens, i+2) == "فروشی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دارند"
            and _tok(tokens, i-1) == "دوست"
            and _tok(tokens, i-2) == "هم"
            and _tok(tokens, i-3) == "خودشان"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1) == "یه‌خورده"
            and _tok(tokens, i+2) == "وقتش"
            and _tok(tokens, i+3) == "گذشته"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "یک"
            and _tok(tokens, i+4) == "درخت"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالایش"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "این"
            and _tok(tokens, i+3) == "انگاری"
            and _tok(tokens, i+4) == "رادیو"
            and _tok(tokens, i+5) == "است"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "را"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+1) == "بگیرد"
        ):
            _set_pos(t, "PART")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "دیگرش"
            and _tok(tokens, i-1) == "سمت"
            and _tok(tokens, i-2) == "بعدا"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "شعری"
            and _tok(tokens, i-1) == "گفته‌های"
            and _tok(tokens, i-2) == "به"
            and _tok(tokens, i+1) == "که"
            and _tok(tokens, i+2) == "آن"
            and _tok(tokens, i+3) == "می‌خواند"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"کتابخانه‌اش","کتابخانهاش"}
        ):
            _set_lemma(t, "کتابخانه")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1) in {"دستهایش","دست‌هایش"}
            and _tok(tokens, i+1) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i-2) == "برود"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "بگیرد"
        ):
            _set_token(t,"بالا")
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+1) == "سمت"
            and _tok(tokens, i+2) == "راست"
        ):
            _set_token(t,"بالا")
            _set_pos(t, "ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قایق"
            and _tok(tokens, i+1) == "درونش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "داستان"
            and _tok(tokens, i+1) == "تعریف"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "قایق"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "توپشون"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پرت"
            and _tok(tokens, i-1) == "توپشان"
            and _tok(tokens, i-1) == "می‌شود"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "شاخه"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "بوده"
            and _tok(tokens, i+1) == "این"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "یارو"
            and _tok(tokens, i-1) == "توت"
            and _tok(tokens, i-2) == "حالا"
            and _tok(tokens, i-3) == "گیلاس"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"سه‌تا","سهتا"}
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i-1) == "این"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "بس"
            and _tok(tokens, i+2) == "هست"
            and _tok(tokens, i+3) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "هست"
            and _tok(tokens, i-2) == "بس"
            and _tok(tokens, i-3) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگری"
            and _tok(tokens, i-1) == "چیز"
            and _tok(tokens, i-2) == "دیگر"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1) == "از"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "خب"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "من"
            and _tok(tokens, i-2) == "والا"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگری"
            and _tok(tokens, i+1) == "نمی‌بینم"
            and _tok(tokens, i+2) == "."
            and _tok(tokens, i-1) == "چیز"
            and _tok(tokens, i-2) == "دیگر"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "وارد"
            and _tok(tokens, i+2) == "شده"
            and _tok(tokens, i-1) == "از"
        ):
            _set_pos(t, "NOUN")
            continue


        if (
            _canon(_tok(tokens, i)) == "باز"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+1) == "رودخانه"
            and _tok(tokens, i+2) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"
            and _tok(tokens, i+1) == "گربه‌ای"
            and _tok(tokens, i+2) == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "گربه"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "حالت"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i-2) == "سگ"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1) == "درخت"
            and _tok(tokens, i-1) == "سگ"
            and _tok(tokens, i+2) == "بوده"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "داداش"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کلاس"
            and _tok(tokens, i+1) == "اول"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"یک‌جور","یکجور"}
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "است"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1) == "سمت"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "والیبال"
        ):
            _set_pos(t, "PRON")
            continue

        

        if (
            _canon(_tok(tokens, i)) == "دوتا"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "اینجا"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "یکی"
            and _tok(tokens, i-1) == "آن"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "جنتلمن"
            and _tok(tokens, i-1) == "خیلی"
            and _tok(tokens, i+1) == "شیک"
            
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "نفر"
            and _tok(tokens, i+1) == "سوار"
            and _tok(tokens, i+2) == "الاغی"
            and _tok(tokens, i+3) == "است"
            
        ):
            _unmark_ez(t)
            continue
        
        if (
            _canon(_tok(tokens, i)) in {"بهمخورده","بهم‌خورده"}
            and _tok(tokens, i+1) == "نیست"
            
        ):
            _set_pos(t, "ADJ")
            continue
        if (
            _canon(_tok(tokens, i)) == "دوتا"
            and _tok(tokens, i+1) == "است"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "می‌گفتند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چیش"
            and _tok(tokens, i-1) == "زیری"
            and _tok(tokens, i-2) == "ظرف"
            and _tok(tokens, i+1) == "را"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "گفتنش"
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i+2) == "می‌خواهم"
        ):
            _set_pos(t, "NOUN")
            _set_lemma(t,"گفتن")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالایش"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیرش"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "میز"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "تعقیبش"
            and _tok(tokens, i+1) == "می‌کند"
            and _tok(tokens, i-1) == "این"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"
            and _lemma(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "می‌بیند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حالت"
            and _tok(tokens, i+1) == "فلزی"
            and _tok(tokens, i+2) == "مشبک"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حالت"
            and _tok(tokens, i+1) == "سنتی"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)

            continue
        if (
            _canon(_tok(tokens, i)) == "بازارهای"
            and _tok(tokens, i+1) == "سرپوشیده"
            and _tok(tokens, i+2) == "قدیمی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزهای"
            and _tok(tokens, i+1) == "سنتی"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شهر"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بازارهای"
            and _tok(tokens, i+1) == "سرپوشیده"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شیشه"
            and _tok(tokens, i+1) == "رنگی"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "میز"
            and _tok(tokens, i+1) == "کوچولویی"
            and _tok(tokens, i+2) in {"هست","است"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سیگار"
            and _tok(tokens, i+1) == "دست"
            and _tok(tokens, i+2) == "او"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیز"
            and _tok(tokens, i+1) == "پخش"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i+2) == "می‌بیند"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) in {"دیگه‌ش","دیگهش"}
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i-1) == "بشقاب‌های"
        ):
            _set_pos(t,"ADJ")
            continue


        if (
            _canon(_tok(tokens, i)) in {"پنکه‌های","پنکههای"}
            and _tok(tokens, i+1) == "پایه‌دار"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ظرف‌های","ظرفهای"}
            and _tok(tokens, i+1) in {"مسی","پایه‌داری","پایه‌دار"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "بگذار"
            and _tok(tokens, i+2) == "ببینم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1) == "مدل"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "اینجا"
            and _tok(tokens, i+2) == "هم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "حیاط"
        ):
            _set_pos(t, "ADP")
            continue


        if (
            _canon(_tok(tokens, i)) == "راستی"
            and _tok(tokens, i+1) == "چون"
            and _tok(tokens, i-1) == "ظاهرا"
            and _tok(tokens, i+2) == "سبک"
            and _tok(tokens, i+3) == "باشد"
            and _tok(tokens, i+4) == "دستشان"
            and _tok(tokens, i+5) == "له"
            and _tok(tokens, i+6) == "می‌شود"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "صورتی"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i-2) == "به"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1) == "کتاب"
            and _tok(tokens, i+2) == "می‌خواند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مادر"
            and _tok(tokens, i+1) == "بافتنی"
            and _tok(tokens, i+2) == "می‌بافد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سیگار"
            and _tok(tokens, i+1) == "کشیدن"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "نمی‌بینم"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "حیاطشون"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "از"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+2) == "این"
            and _tok(tokens, i+3) == "طرف"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i-1) == "سگ"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i-1) == "راست"
            and _tok(tokens, i+2) == "می‌کند"
            and _tok(tokens, i-2) == "سمت"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) == "خانه"
            and _tok(tokens, i-1) == "احتمال"
            and _tok(tokens, i+2) == "خودشان"
            and _tok(tokens, i-2) == "که"
            and _tok(tokens, i+3) == "باشد"
        ):
            _set_pos(t,"VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلو"
            and _tok(tokens, i-1) == "الاغی"
            and _tok(tokens, i-2) == "سوار"

        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "توپ"
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i+2) == "از"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اسکله"
            and _tok(tokens, i+1) == "مانندی"
            and _tok(tokens, i-1) == "پل"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بیل"
            and _tok(tokens, i+1) == "ماسه"
            and _tok(tokens, i-1) == "با"
            and _tok(tokens, i+2) == "درست"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ماسه"
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "راست"
            and _tok(tokens, i-1) == "سمت"
            and _tok(tokens, i+1) == "دنبال"
            and _tok(tokens, i+2) == "گربه"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "هوا"
            and _tok(tokens, i-1) == "تهویه"
            and _tok(tokens, i+1) == "وجود"
            and _tok(tokens, i+2) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ظروف"
            and _tok(tokens, i-1) == "،"
            and _tok(tokens, i+1) == "مسی"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"نوشابه‌ای","نوشابهای","دریاچه‌ای","دریاچهای"}
            and _tok(tokens, i+1) == "چیزی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"شش‌تا","ششتا"}
        ):
            _set_pos(t, "NUM")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"سه‌تا","سهتا"}
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) in {"دوتا","دو‌تا"}
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌ها","اینها","این‌","‌این"}
        ):
            _set_lemma(t, "این")
            continue

        if (
            _canon(_tok(tokens, i)) in {"میوهجات","میوه‌جات"}
        ):
            _set_lemma(t, "میوه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"به‌اضافهی","بهاضافهی","به‌اضافه‌ی"}
        ):
            _set_lemma(t, "به‌اضافه")
            continue
        if (
            _canon(_tok(tokens, i)) in {"بچه‌ش","بچهش"}
        ):
            _set_lemma(t, "بچه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"همه‌اش","همهاش"}
        ):
            _set_lemma(t, "همه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"چهارپایه‌اش","چهارپایهاش"}
        ):
            _set_lemma(t, "چهارپایه")
            continue
        if (
            _canon(_tok(tokens, i)) in {"پرده‌اش","پردهاش"}
        ):
            _set_lemma(t, "پرده")
            continue
        if (
            _canon(_tok(tokens, i)) in {"خانه‌اش","خانهاش"}
        ):
            _set_lemma(t, "خانه")
            continue
        if (
            _canon(_tok(tokens, i)) in {"خانه‌شان","خانهشان"}
        ):
            _set_lemma(t, "خانه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"کتابخونه‌اش","کتابخونهاش"}
        ):
            _set_lemma(t, "کتابخانه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"بقیهاش","بقیه‌اش"}
        ):
            _set_lemma(t, "بقیه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"برگهایش","برگ‌هایش","برگ‌ها‌یش","برگ‌ها"}
        ):
            _set_lemma(t, "برگ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"همین‌ها","همین"}
        ):
            _set_lemma(t, "همین")
            continue

        if (
            _canon(_tok(tokens, i)) in {"اون‌ها","اونها"}
        ):
            _set_lemma(t, "آن")
            continue

        if (
            _canon(_tok(tokens, i)) in {"بچه‌شون","بچهشون","بچه‌شان","بچهشان"}
        ):
            _set_lemma(t, "بچه")
            continue

        if (
            _canon(_tok(tokens, i)) == "نصفش"
            and _tok(tokens, i+1) == "مال"
            and _tok(tokens, i+2) == "شما" 
        ):
            _set_lemma(t, "نصف")
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+2) == "درون" 
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "ها"
            and _tok(tokens, i-1) == "است"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "فکر"
            and _tok(tokens, i+3) == "می‌کنم"
            and _tok(tokens, i+4) == "یک"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "منزل"
            and _tok(tokens, i+3) == "است"
            and _tok(tokens, i-1) == "دقیقا"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "زمانی"
            and _tok(tokens, i-1) in {"آقای","اقای"}

        ):
            _set_pos(t, "PROPN")
            _set_lemma(t, "زمانی")
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرفش","آنطرفش"}
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "کوچیک‌تر"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"زیبایی", "زیبای"}    # tolerate both
            and _canon(_tok(tokens, i+1)) in {"طاقهای", "طاق‌های"}  # tolerate ZWNJ
        ):
            _mark_ez(tokens[i])
            _mark_ez(t)

        if (
            _tok(tokens, i) == "دیگر"
            and _tok(tokens, i-1) == "دقیقا"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "منزل"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _tok(tokens, i) == "شوهر"
            and _tok(tokens, i+1) == "او"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "پسر"
            and _tok(tokens, i+1) == "کوچک‌تر"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "چیزهای"
            and _tok(tokens, i+1) == "آهنی"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "محل"
            and _tok(tokens, i+1) == "تفریحی"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "مغازه"
            and _tok(tokens, i+1) == "کوچولو"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "صدای"
            and _tok(tokens, i+1) == "این‌ها"
        ):
            _mark_ez(t)
            continue
        

        if (
            _tok(tokens, i) == "بیرون"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "منظره"
        ):
            _set_pos(t,"NOUN")
            continue
        
        if (
            _tok(tokens, i) == "دیگر"
            and _tok(tokens, i+1) == "نشان"
            and _tok(tokens, i-1) == "می‌دهد"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _tok(tokens, i) == "ساحل"
            and _tok(tokens, i+1) == "وجود"
            and _tok(tokens, i-1) == "کنار"
            and _tok(tokens, i+2) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) == "گرفته"
            and _tok(tokens, i-1) == "کمرش"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) == "می‌کنند"
            and _tok(tokens, i-1) == "حالش"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) == "اگر"
            and _tok(tokens, i-1) == "این‌ها"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) == "می‌آید"
            and _tok(tokens, i-1) == "آن"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) == "میل"
            and _tok(tokens, i-1) == "چیزی"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) == "پارک"
            and _tok(tokens, i-1) == "اتومبیلشون"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) == "آن‌طوری"
            and _tok(tokens, i-1) == "دست‌هایش"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) == "جمع"
            and _tok(tokens, i-1) == "پاهایش"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) in {"بگو","درست","بگیرد","دیگر","بنشینی","دارد","درون"}
            and _tok(tokens, i-1) in {"این","اینجا","این‌ها","دست","آن"}
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) == "پشت"
            and _tok(tokens, i+1) == "دیوار"
            and _tok(tokens, i+2) == "."
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "مامان"
            and _tok(tokens, i+1) == "حواسش"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i+1) in {"نمی‌بیند","ببینم"}
            and _tok(tokens, i-1) == "آن"
        ):
            _set_pos(t,"ADP")
            continue
        


        if (
            _tok(tokens, i) == "را"
            and _tok(tokens, i-1) == "دست‌هایشان"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _tok(tokens, i) in {"آنطرفش","آن‌طرفش"}
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i-1) == "در"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _tok(tokens, i) == "محل"
            and _tok(tokens, i+1) == "استراحتی"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue


        if (
            _tok(tokens, i) == "بیرون"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "منظره"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _tok(tokens, i) == "خوب"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "سقف"
        ):
            _set_pos(t,"INTJ")
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "تصویر"
            and _tok(tokens, i+1) == "بعد"
            and _tok(tokens, i+2) == "سقف"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "دیگر"
            and _tok(tokens, i-1) == "خوب"
            and _tok(tokens, i+1) == "سقف"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "بعد"
            and _tok(tokens, i-1) == "تصویر"
        ):
            _set_pos(t,"NOUN")
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) in {"آشپزخانه‌اش","آشپزخانهاش"}
        ):
            _set_lemma(t, "آشپزخانه")
            continue

        if (
            _tok(tokens, i) in {"آنها","آن‌ها"}
        ):
            _set_lemma(t, "آن")
            continue

        if (
            _tok(tokens, i) in {"همین‌ها","همینها"}
        ):
            _set_lemma(t, "همین")
            continue

        if (
            _tok(tokens, i) == "هرکدومش"
        ):
            _set_lemma(t, "هرکدام")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) in {"صندلی","آب"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1)  == "چیز"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1)  == "من"
            and _tok(tokens, i+1)  == "پیر"
            and _tok(tokens, i+2)  == "شدم"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1)  == "این‌ها"
            and _tok(tokens, i+1)  == "سه‌تا"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "کابینت"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "باز"
            and _tok(tokens, i+4)  == "کرده"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "ظرف"
            and _tok(tokens, i+2)  == "کلوچه"
            and _tok(tokens, i+3)  == "را"
            and _tok(tokens, i+4)  == "برداشته"
        ):
            _set_pos(t,"NOUN")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دریا"
            and _tok(tokens, i-1)  == "کنار"
            and _tok(tokens, i-2)  == "مثل"
            and _tok(tokens, i+3)  == "بچه‌ها"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مثل"
            and _tok(tokens, i+1)  == "کنار"
            and _tok(tokens, i+2)  == "دریا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کودک"
            and _tok(tokens, i+1)  == "کلاه"
            and _tok(tokens, i+2)  == "سنتی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درختی"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i+2)  == "چیده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "پنکه‌های"
            and _tok(tokens, i+2)  == "پایه‌دار"
            and _tok(tokens, i+3)  == "است"
        ):
            _set_pos(t,"DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "نمی‌دانم"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "مغازه‌ها"
            and _tok(tokens, i-1)  == "این"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیرشان"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "که"
            and _tok(tokens, i+3)  == "فکر"
            and _tok(tokens, i+4)  == "می‌کنم"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "قایق"
            and _tok(tokens, i+1)  == "چیزها"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i-2)  == "از"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روز"
            and _tok(tokens, i+1)  == "حالش"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "می‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"دستدوز","دست‌دوز"}
            and _tok(tokens, i+1)  == "سرش"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سقف"
            and _tok(tokens, i-1)  == "بالای"
            and _tok(tokens, i+1)  == "وجود"
            and _tok(tokens, i+2)  == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اولش"
            and _tok(tokens, i-1)  == "ببین"
            and _tok(tokens, i+1)  == "بالای"
            and _tok(tokens, i+2)  == "یک"
            and _tok(tokens, i+3)  == "درخت"
        ):
            _set_pos(t,"NOUN")
            _set_lemma(t,"اول")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبه‌روی","روبهروی"}
            and _tok(tokens, i+1)  in {"حیاطشون","حیاطشان"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"همهی","همه‌ی"}
            and _tok(tokens, i+1)  == "مخلفاتش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "قرآن"
            and _tok(tokens, i+2)  == "دست"
        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "استاد"
            and _tok(tokens, i+1)  == "نقاشی"
            and _tok(tokens, i+2)  == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i-1)  == "نشسته‌اند"
            and _tok(tokens, i+2)  == "میزگرد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "چه"
            and _tok(tokens, i+3)  == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "گاراژ"
            and _tok(tokens, i+2)  == "هم"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "سر"
            and _tok(tokens, i+2)  == "اون‌ها"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1)  == "کنار"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "اینجا"
        ):
            _set_pos(t,"PRON")
            continue
        
        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "جدا"
            and _tok(tokens, i+3)  == "کردند"

        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "سفیدکاری"
            and _tok(tokens, i+3)  == "بکند"

        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "این"
            and _tok(tokens, i+3)  == "چیز"

        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "آب"
            and _tok(tokens, i+3)  == "است"

        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "،"
            and _tok(tokens, i+3)  == "بعد"

        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "فرار"
            and _tok(tokens, i+3)  == "می‌کند"

        ):
            _set_pos(t,"PRON")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "نصف"
            and _tok(tokens, i+3)  == "که"
            and _tok(tokens, i+4)  == "رفته"

        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1)  == "می‌خواسته"
            and _tok(tokens, i+1)  == "،"
            and _tok(tokens, i+2)  == "حالا"
            and _tok(tokens, i+3)  == "میوه"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "آن"
            and _tok(tokens, i+3)  == "طرف"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "باز"
            and _tok(tokens, i+3)  == "ظاهرا"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i-2)  == "می‌برد"
            and _tok(tokens, i-3)  == "پناه"

        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1)  == "تو"
            and _tok(tokens, i+1)  == "قفسه"
            and _tok(tokens, i+2)  == "چه"
            and _tok(tokens, i+3)  == "گذاشته‌اند"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "آبی"
            and _tok(tokens, i+3)  == "ایستاده"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دم"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "میز"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پارکی"
            and _tok(tokens, i+1)  == "جایی"
            and _tok(tokens, i-1)  == "تو"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i+1)  == "بچه"
            and _tok(tokens, i-1)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "نفر"
            and _tok(tokens, i-3)  == "دو"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1) in {"بکند","می‌گیرد"}
        ):
            _set_pos(t,"AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i-1)  == "فرش"
            and _tok(tokens, i+1)  == "چرخ"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "چرخ"
            and _tok(tokens, i+2)  == "گذاشته"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "الاغ"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "توجه"
            and _tok(tokens, i-1)  == "قابل"
            and _tok(tokens, i+1)  == "دیگری"
            and _tok(tokens, i+2)  == "نیست"
        ):
            _mark_ez(t)
            
            
        if (
            _canon(_tok(tokens, i)) == "قدیمی"
            and _tok(tokens, i+1)  == "دیگر"
            and _tok(tokens, i+2)  == "."
        ):
            _set_pos(t,"ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"منتهای‌مراتب","منتهایمراتب"}
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جا"
            and _tok(tokens, i+1)  == "پیدا"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"پنج‌تا","پنجتا"}
        ):
            _set_pos(t,"NUM")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بین"
            and _tok(tokens, i+1)  == "زمین"
            and _tok(tokens, i+2)  == "و"
            and _tok(tokens, i+3)  == "آسمان"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دخترخانم"
            and _tok(tokens, i+1)  == "جوانی"
            and _tok(tokens, i-1)  == "یک"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i+1)  == "باز"
            and _tok(tokens, i+2)  == "کردند"
        ):
            _unmark_ez(t)
            continue

        
        if (
            _tok(tokens, i) == "ه‌مان‌طور"
        ):
            _set_token(t, "همان‌طور")
            continue

        if (
            _canon(_tok(tokens, i)) == "داخل"
            and _tok(tokens, i+1)  == "بازار"
            and _tok(tokens, i-1)  == "تو"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "همان"
            and _tok(tokens, i+2)  == "آقا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "لای"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "درخت‌ها"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقای"
            and _tok(tokens, i+1)  == "دیگری"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قسمت"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "برجسته"
            and _tok(tokens, i+2)  == "نامشخص"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیز"
            and _tok(tokens, i+1)  == "دیگری"
            and _tok(tokens, i+2)  == "نیست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قایق"
            and _tok(tokens, i+1)  == "پارویی"
            and _tok(tokens, i+2)  == "بوده"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ساختمان"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "قدیمی"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مستطیل"
            and _tok(tokens, i-1)  == "همان"
            and _tok(tokens, i+1)  == "اول"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جای"
            and _tok(tokens, i+1)  == "ساحلی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالاییه"
            and _tok(tokens, i-1)  == "آقا"
            and _tok(tokens, i-2)  == "آن"
        ):
            _set_pos(t,"ADJ")
            _set_lemma(t,"بالا")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "همان"
            and _tok(tokens, i+2)  == "بازارچه"
        ):
            _mark_ez(t)
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i+1)  == "داخل"
            and _tok(tokens, i+2)  == "بازار"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1)  == "سرش"
            and _tok(tokens, i-1)  == "گذاشته"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "آب"
            and _tok(tokens, i+3)  == "ایستاده"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "مبلی"
            and _tok(tokens, i+2)  == "نشسته"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بین"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "خانم"
            and _tok(tokens, i+3)  == "و"
            and _tok(tokens, i+4)  == "آقا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "تلویزیون"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "احتمالا"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "سفره‌ها"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "مال"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "دریا"
            and _tok(tokens, i-1)  == "کنار"
            and _tok(tokens, i-2)  == "در"
            and _tok(tokens, i+1)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "دیگری"
            and _tok(tokens, i-1)  == "خاص"
            and _tok(tokens, i-2)  == "چیز"
            and _tok(tokens, i+1)  == "دیده"
            and _tok(tokens, i+2)  == "نمی‌شود"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _tok(tokens, i) == "خاص"
            and _tok(tokens, i-1)  == "چیز"
            and _tok(tokens, i+1)  == "دیگری"
        ):
            _mark_ez(t)
            continue

        if (
            _tok(tokens, i) == "پارو"
            and _tok(tokens, i-1)  == "با"
            and _tok(tokens, i+1)  == "حرکت"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        

        if (
            _canon(_tok(tokens, i)) == "قرآن"
            and _tok(tokens, i+1)  == "دست"
            and _tok(tokens, i+2)  == "او"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرف","آنطرف"}
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "فضایی"
            and _tok(tokens, i+3)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i-1)  == "هست"
            and _tok(tokens, i+1)  == "،"
            and _tok(tokens, i-2)  == "چه"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "باز"
            and _tok(tokens, i+1)  == "دارند"
            and _tok(tokens, i+2)  == "توپ‌بازی"
            and _tok(tokens, i+3)  == "می‌کنند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"جابه‌جا","جابهجا"}
            and _tok(tokens, i+1)  == "می‌کند"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "فرش"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "می‌گویند"
            and _tok(tokens, i+1)  == "بکنیم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "خوابیده"
            and _tok(tokens, i-1)  == "حالت"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "خودتان"
            and _tok(tokens, i+1)  == "مخلفاتش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بادبادک"
            and _tok(tokens, i+1)  == "هوا"
            and _tok(tokens, i-1)  == "حال"
            and _tok(tokens, i+2)  == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"پردههای","پرده‌های"}
            and _tok(tokens, i+1)  == "آن"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1)  == "طبق"
            and _tok(tokens, i+2)  == "آن"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"بچهها","بچه‌ها"}
            and _tok(tokens, i+1)  == "سوار"
            and _tok(tokens, i+2)  == "قایق"
            and _tok(tokens, i+3)  == "شده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"همه‌ی","همهی"}
        ):
            _set_lemma(t,"همه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"سه‌پایه‌اش","سهپایهاش"}
        ):
            _set_lemma(t,"سه‌پایه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"شانه‌اش","شانهاش"}
        ):
            _set_lemma(t,"شانه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روی","روی‌"}
        ):
            _set_lemma(t,"روی")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "هست"
            and _tok(tokens, i-1)  == "رینگ"
            and _tok(tokens, i-2)  == "روی"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "منهای"
            and _tok(tokens, i+1)  == "تابلوی"
            and _tok(tokens, i+2)  == "وسطی"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "رینگ"
            and _tok(tokens, i+2)  == "در"
            and _tok(tokens, i+3)  == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "رینگ"
            and _tok(tokens, i+1)  == "در"
            and _tok(tokens, i+2)  == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانواده"
            and _tok(tokens, i+1)  == "داخلش"
            and _tok(tokens, i+2)  == "هستند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1)  == "خشک"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "نمی‌دهم"
            and _tok(tokens, i-2)  == "تشخیص"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "می‌خواهد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چپق"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "گرفته"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i+1)  == "چپق"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i-2)  == "با"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "موقع"
            and _tok(tokens, i+1)  == "یادم"
            and _tok(tokens, i-1)  == "هیچ"
            and _tok(tokens, i+2)  == "نمی‌رود"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ماشین"
            and _tok(tokens, i+1)  == "پارک"
            and _tok(tokens, i+2)  == "است"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پارک"
            and _tok(tokens, i+1)  == "است"
            and _tok(tokens, i-1)  == "ماشین"

        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "منهای"
            and _tok(tokens, i-1)  == "نمی‌شود"
            and _tok(tokens, i-2)  == "دیده"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1)  == "در"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "مغازه"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "نزدیکشان"
            and _tok(tokens, i-1)  == "رادیو"
            and _tok(tokens, i-2)  == "یک"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == ""
            and _tok(tokens, i-1)  == "رادیو"
            and _tok(tokens, i-2)  == "یک"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "طبیعت"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i-2)  == "در"
            and _tok(tokens, i+1)  == "گربه‌ای"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کشیدن"
            and _tok(tokens, i-1)  == "سیگار"
            and _tok(tokens, i+1)  == "هست"
        ):
            _set_pos(t,"VRB")
            continue

        if (
            _canon(_tok(tokens, i)) == "دوتاش"
        ):
            _set_lemma(t,"دوتا")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "به"
            and _tok(tokens, i+1)  == "نزدیک"
            and _tok(tokens, i+2)  == "می‌شود"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "نزدیک"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "می‌شود"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بیشتر"
            and _tok(tokens, i+1)  == "چون"
            and _tok(tokens, i+2)  == "آب"
            and _tok(tokens, i+3)  == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دریا"
            and _tok(tokens, i+1)  == "بیشتر"
            and _tok(tokens, i-1)  == "کنار"
            and _tok(tokens, i+2)  == "چون"
            and _tok(tokens, i+3)  == "آب"
            and _tok(tokens, i+4)  == "است"
        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1)  == "پدر"
            and _tok(tokens, i+2)  == "و"
            and _tok(tokens, i+3)  == "مادر"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کار"
            and _tok(tokens, i+1)  == "انجام"
            and _tok(tokens, i+2)  == "بدهند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"چرخ‌دستی","چرخدستی"}
            and _tok(tokens, i+1)  == "را"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "تلویزیون"
            and _tok(tokens, i+1)  == "تماشا"
            and _tok(tokens, i+2)  == "کردند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کلبه"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+2)  == "ماشین"
            and _tok(tokens, i-2)  == "تو"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کابینت"
            and _tok(tokens, i+1)  == "دوتا"
            and _tok(tokens, i-1)  == "روی"
            and _tok(tokens, i+2)  == "فنجان"
            and _tok(tokens, i-2)  == "بعد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شنی"
            and _tok(tokens, i+1)  == "درست"
            and _tok(tokens, i-1)  == "قلعه"
            and _tok(tokens, i+2)  in {"می‌کند","می‌کنند"}
            and _tok(tokens, i-2)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کناری"
            and _tok(tokens, i-1)  == "تابلوی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "گربه"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "حرکت"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i+1)  == "گذاشته"
            and _tok(tokens, i-2)  == "کلوچه"
        ):
            _set_pos(t, "NOUN")
            continue
        
        if (
            _canon(_tok(tokens, i)) == "وسطی"
            and _tok(tokens, i-1)  == "تابلوی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سطلی"
            and _tok(tokens, i+1)  == "دست"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+2)  == "او"
            and _tok(tokens, i+3)  == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"چرخدستی","چرخدستی"}
            and _tok(tokens, i+1)  == "فرش"
            and _tok(tokens, i-1)  == "با"
            and _tok(tokens, i+2)  == "حمل"
            and _tok(tokens, i+3)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فرش"
            and _tok(tokens, i+1)  == "حمل"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"شش‌تا","ششتا"}
            and _tok(tokens, i+1)  == "تصویر"
            and _tok(tokens, i-1)  == "این"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقاپسرها"
            and _tok(tokens, i+1)  == "سعی"
            and _tok(tokens, i+2)  == "می‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اتاق"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "پنجره"
            and _tok(tokens, i+3)  == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ساعت"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "کتابخونه‌ای"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i-2)  == "زیر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تلویزیون"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "گربه‌ای"
            and _tok(tokens, i-1)  == "جلوی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مادر"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "ظرف"
            and _tok(tokens, i+3)  == "میوه"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "دیوار"
            and _tok(tokens, i+2)  == "شروع"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "پارکینگ"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) in {"بچهها","بچه‌ها"}
            and _tok(tokens, i-2)  == "یکی"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i+1)  == "کوچک‌تر"
            and _tok(tokens, i+2)  == "از"
            and _tok(tokens, i+3)  == "بقیه"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"مغازه‌ها","مغازهها"}
            and _tok(tokens, i-2)  == "دو"
            and _tok(tokens, i-1)  == "سمتش"
            and _tok(tokens, i+1)  == "قرار"
            and _tok(tokens, i+2)  == "دارند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"میوه‌ی","میوهی"}
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "لب"
            and _tok(tokens, i+3)  == "ساحل"
            and _tok(tokens, i+4)  == "را"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "حالا"
            and _tok(tokens, i+3)  == "یک"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "سنتی"
            and _tok(tokens, i+1)  == "است"
            and _tok(tokens, i-1)  == "درواقع"
            and _tok(tokens, i-2)  == "بازار"
        ):
            _set_pos(t, "ADJ")
            _set_lemma(t, "سنتی")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "منافذی"
            and _tok(tokens, i-1)  == "از"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "توپ"
            and _tok(tokens, i-1)  == "می‌روند"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) in {"خاطره‌انگیزی","خاطرهانگیزی"}
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبه‌رو","روبهرو"}
            and _tok(tokens, i+1)  == "می‌آیند"
            and _tok(tokens, i-1)  == "از"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"مو","موی"}
            and _tok(tokens, i+1)  in {"سفید","سفیدی"}
            and _tok(tokens, i-1)  == "پیرمرد"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "به"
            and _tok(tokens, i+1)  == "شب"
            and _tok(tokens, i-1)  == "بیشتر"
            and _tok(tokens, i+2)  == "یلدا"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "بشقابی"
            and _tok(tokens, i+3)  == "را"
            and _tok(tokens, i+4)  == "تمیز"
            and _tok(tokens, i+5)  == "می‌کند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) in {"سایه‌اش","سایهاش"}
        ):
            _set_lemma(t, "سایه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"پنکه‌شان","پنکهشان"}
        ):
            _set_lemma(t, "پنکه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"سنگ‌هایی","سنگهایی"}
        ):
            _set_lemma(t, "سنگ")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1) == "گربه"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _set_pos(t, "NOUN")
            continue


        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1) == "از"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "وارد"
            and _tok(tokens, i+3) == "اتاق"
            and _tok(tokens, i+4) == "می‌شود"
        ):
            _set_pos(t, "NOUN")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "به"
            and _tok(tokens, i-1) == "مثلا"
            and _tok(tokens, i+1) == "نزدیک"
            and _tok(tokens, i+2) == "می‌شود"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1) == "کنار"
            and _tok(tokens, i+1) == "آویزان"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i-2) == "که"
            and _tok(tokens, i+1) == "هست"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "ظاهرا"
            and _tok(tokens, i+2) == "سمنویی"
            and _tok(tokens, i+3) == "است"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "پارو"
            and _tok(tokens, i-1) == "با"
            and _tok(tokens, i+1) == "حرکت"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i+1) == "چپش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1) == "کنار"
            and _tok(tokens, i+1) == "آویزان"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "در"
            and _tok(tokens, i+2) == "آویزان"
            and _tok(tokens, i+3) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فضای"
            and _tok(tokens, i+1) == "آزاد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "پا"
            and _tok(tokens, i+2) == "وایمیسادیم"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "آن"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "کتابخونه‌ای"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "میز"
            and _tok(tokens, i+3) == "نشسته‌اند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "خانم"
            and _tok(tokens, i+2) == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "پای"
            and _tok(tokens, i+2) == "خانم"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "سایه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "صندلی‌های"
            and _tok(tokens, i+2) == "عقب"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "باعث"
            and _tok(tokens, i+1) == "روشن"
            and _tok(tokens, i+2) == "شدن"
            and _tok(tokens, i+3) == "محوطه‌ی"
            and _tok(tokens, i+4) == "بازار"
            and _tok(tokens, i+5) == "شده"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "پرچمی"
            and _tok(tokens, i-3) == "یک"
            and _tok(tokens, i+2) == "کلبه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1) == "نردبان"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "معروف"
            and _tok(tokens, i-1) == "قول"
            and _tok(tokens, i-2) == "به"
            and _tok(tokens, i+1) == "پیک‌نیکی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شربتی"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "می‌خواهد"
            and _tok(tokens, i+3) == "بخورد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیردست"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "آقا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "همراه"
            and _tok(tokens, i+1) == "با"
            and _tok(tokens, i+2) == "آقای"
            and _tok(tokens, i+3) == "پیر"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "همراه"
            and _tok(tokens, i+1) == "با"
            and _tok(tokens, i+2) == "آقایی"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "برعکس"
            and _tok(tokens, i+1) == "کلاه"
            and _tok(tokens, i+2) == "سه"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i+1) == "درون"
            and _tok(tokens, i+2) == "درخت"
            and _tok(tokens, i+3) == "گیر"
            and _tok(tokens, i+4) == "می‌کند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "برگ"
            and _tok(tokens, i-1) == "از"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "نوشتن"
            and _tok(tokens, i+1) == "به"
            and _tok(tokens, i+2) == "درد"
            and _tok(tokens, i+2) == "ما"
            and _tok(tokens, i+2) == "می‌خورد"
        ):
            _set_pos(t, "VERB")
            continue

        if (
            _canon(_tok(tokens, i)) in {"بدون‌اینکه","بدوناینکه"}
        ):
            _set_pos(t, "SCONJ")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "پسربچه‌ای"
            and _tok(tokens, i+3) == "نشسته"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیاد"
            and _tok(tokens, i+1) == "آشپزخانه"
            and _tok(tokens, i+2) == "است"
            and _tok(tokens, i-1) == "احتمال"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i+1) == "توپ"
            and _tok(tokens, i+2) == "دستش"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سطلی"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
            and _tok(tokens, i-1) == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مس"
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i-1) == "جنس"
            and _tok(tokens, i-2) == "از"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پارچه"
            and _tok(tokens, i+1) == "آویزان"
            and _tok(tokens, i+2) == "کرده"
            and _tok(tokens, i-1) == "مقدار"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مقدار"
            and _tok(tokens, i+1) == "پارچه"
            and _tok(tokens, i+2) == "آویزان"
            and _tok(tokens, i+3) == "کرده"
            and _tok(tokens, i-1) == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"می‌اورد","میاورد"}
        ):
            _set_token(t,"می‌آورد")
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویری"
            and _tok(tokens, i-1) == "چه"
            and _tok(tokens, i+1) == "،"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پیر"
            and _tok(tokens, i-1) == "آقای"
            and _tok(tokens, i+1) == "سوار"
        ):
            _set_pos(t, "ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "الاغی"
            and _tok(tokens, i-1) == "سوار"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانواده"
            and _tok(tokens, i-1) == "پدر"
            and _tok(tokens, i+1) == "حضور"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "آشپزخانه"
            and _tok(tokens, i+3) == "را"
            and _tok(tokens, i+4) == "نشان"
            and _tok(tokens, i+5) == "می‌دهد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "دریا"
            and _tok(tokens, i+3) == "یا"
            and _tok(tokens, i+4) == "دریا"
            and _tok(tokens, i+5) == "یا"
            and _tok(tokens, i+6) == "رودخانه"
            and _tok(tokens, i+7) == "یا"
            and _tok(tokens, i+8) == "دریاچه"
            and _tok(tokens, i+9) == "را"
            and _tok(tokens, i+10) == "نشان"
            and _tok(tokens, i+11) == "می‌دهد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دریا"
            and _tok(tokens, i-1) == "کنار"
        ):
            _unmark_ez(t)
            continue

        if (
            _tok(tokens, i) == "صندلی‌های"
            and _tok(tokens, i+1) == "عقب"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "به"
            and _tok(tokens, i-1) == "بالا"
            and _tok(tokens, i-2) == "از"
            and _tok(tokens, i+1) == "پایین"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "پنجره"
            and _tok(tokens, i+1) == "بخشی"
            and _tok(tokens, i+2) == "از"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سرریز"
            and _tok(tokens, i+1) in {"شدن","کرده"}
        ):
            _set_pos(t,"NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ساحل"
            and _tok(tokens, i+1) == "دست"
            and _tok(tokens, i-1) == "تو"
            and _tok(tokens, i+2) == "می‌گیرند"
        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) in {"مغازهها","مغازه‌ها"}
            and _tok(tokens, i+1) == "دو"
            and _tok(tokens, i+2) == "سمتش"
            and _tok(tokens, i+3) == "قرار"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"مغازهها","مغازه‌ها","مغازه"}
            and _tok(tokens, i+1) == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "نورگیری"
            and _tok(tokens, i+2) == "دارد"
            and _tok(tokens, i-1) == "آن"
        ):
            _set_token(t,"بالا")
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پای"
            and _tok(tokens, i+1) == "سینی"
            and _tok(tokens, i+2) == "هفت‌سین"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پارویی"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-2) == "درون"
            and _tok(tokens, i-1) == "قایق"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "سرریز"
            and _tok(tokens, i+1) == "کرده"
            and _tok(tokens, i-1) == "ظرفشویی"
            and _tok(tokens, i-2) == "سینک"
        ):
            _set_pos(t, "NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سرریز"
            and _tok(tokens, i+1) == "کرده"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i-2) == "ظرفشویی"
            and _tok(tokens, i-3) == "سینک"
        ):
            _set_pos(t, "NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1) == "سینک"
            and _tok(tokens, i+2) == "ظرفشویی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فرد"
            and _tok(tokens, i+1) == "وجود"
            and _tok(tokens, i+2) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "استفاده"
            and _tok(tokens, i+2) == "کرده"
            and _tok(tokens, i-1) == "به"
            and _tok(tokens, i-2) == "پشتش"
        ):
            _set_pos(t, "PRON")
            continue


        if (
            _canon(_tok(tokens, i)) in {"خاطره‌انگیزی","خاطرهانگیزی"}
            and _tok(tokens, i+1) == "از"
            and _tok(tokens, i+2) == "دوران"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i+1) == "نورگیری"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+2) == "دارد"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "پنجره"
            and _tok(tokens, i+1) == "مانندی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فرش"
            and _tok(tokens, i+1) == "فروشان"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلویش"
            and _tok(tokens, i+1) == "قرار"
            and _tok(tokens, i-1) == "که"
            and _tok(tokens, i+2) == "دارد"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلویش"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "پنجره"
            and _tok(tokens, i+3) == "هست"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "جلویش"
            and _tok(tokens, i+1) == "قرار"
            and _tok(tokens, i+2) == "دارد"
            and _tok(tokens, i-1) == "که"
        ):
            _set_pos(t, "NOUN")
            continue



        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1) == "سوار"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "الاغی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اولین"
            and _tok(tokens, i+1) == "روز"
            and _tok(tokens, i+2) == "مدرسه"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "سگ"
            and _tok(tokens, i-1) == "و"
            and _tok(tokens, i+2) == "دنبال"
            and _tok(tokens, i+3) == "گربه"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "مانندی"
            and _tok(tokens, i+1) == "وجود"
            and _tok(tokens, i+2) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1) == "قایق"
            and _tok(tokens, i+2) == "پارویی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تلویزیون"
            and _tok(tokens, i+1) == "تابلویی"
            and _tok(tokens, i+2) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "توضیح"
            and _tok(tokens, i+2) == "تابلو"
            and _tok(tokens, i+3) == "بدهم"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"پنج‌تا","پنجتا"}
            and _tok(tokens, i-1) == "چهارتا"
            and _tok(tokens, i-2) == "چندتا"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "چندتا"
            and _tok(tokens, i+1) == "چهارتا"
            and _tok(tokens, i+2) == "پنج‌تا"
        ):
            _set_pos(t, "NUM")
            _unmark_ez(t)

            continue

        if (
            _canon(_tok(tokens, i)) == {"سه‌ای","سهای"}
            and _tok(tokens, i-1) == "عدد"
            and _tok(tokens, i+1) == "که"
            and _tok(tokens, i+2) == "آنجا"
            and _tok(tokens, i+3) == "دیده"
            and _tok(tokens, i+4) == "می‌شود"
        ):
            _set_pos(t, "ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سرریز"
            and _tok(tokens, i+1) == "شدن"
            and _tok(tokens, i+2) == "آب"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "شدن"
            and _tok(tokens, i-1) == "سرریز"
            and _tok(tokens, i+1) == "آب"
        ):
            _set_pos(t, "VERB")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلودست"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "حرکت"
            and _tok(tokens, i+3) == "می‌کند"
        ):
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"دیگ‌های","دیگهای"}
            and _tok(tokens, i+1) == "مسی"
        ):
            _set_pos(t, "NOUN")
            _set_lemma(t, "دیگ")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اتاق"
            and _tok(tokens, i+1) == "سه‌تا"
            and _tok(tokens, i-1) == "دیوار"
            and _tok(tokens, i+2) == "درواقع"
            and _tok(tokens, i-2) == "روی"
            and _tok(tokens, i+3) == "تابلو"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+2) == "نورگیری"
            and _tok(tokens, i+3) == "دارد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "رفت"
            and _tok(tokens, i+2) == "و"
            and _tok(tokens, i+3) == "آمد"
            and _tok(tokens, i+4) == "می‌کنند"
            and _tok(tokens, i-1) == "در"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "باز"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i-1) == "کنار"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "کلاه"
            and _tok(tokens, i+3) == "به"
            and _tok(tokens, i+4) == "سر"
            and _tok(tokens, i+5) == "دارد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "باز"
            and _tok(tokens, i+3) == "کلاه"
            and _tok(tokens, i+4) == "دارد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "خب"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "باز"
            and _tok(tokens, i+3) == "هم"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) in {"آن","پنجره"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بچه"
            and _tok(tokens, i+1) == "او"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "پای"
            and _tok(tokens, i+2) == "این"
            and _tok(tokens, i+3) == "فرد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "سایه‌اش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) in {"دیوار","پایش","یک","کابینتی"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "مغازه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بین"
            and _tok(tokens, i+1) == "خانم"
            and _tok(tokens, i+2) == "و"
            and _tok(tokens, i+3) == "آن"
            and _tok(tokens, i+4) == "آقای"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "باز"
            and _tok(tokens, i+3) == "یک"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بین"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "در"
            and _tok(tokens, i+3) == "فاصله"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بین"
            and _tok(tokens, i+1) in {"این‌ها","پایش"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "رینگ"
            and _tok(tokens, i+2) == "در"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تابلو"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "تلویزیون"
            and _tok(tokens, i+3) == "به"
            and _tok(tokens, i+4) == "دیوار"
            and _tok(tokens, i+5) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "نگاه"
            and _tok(tokens, i+2) == "می‌کنند"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i-2) == "به"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1) == "در"
            and _tok(tokens, i+2) == "نگاه"
            and _tok(tokens, i+3) == "می‌کنند"
            and _tok(tokens, i-1) == "به"
        ):
            _set_pos(t, "DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "سمت"
            and _tok(tokens, i+1) == "درختی"
            and _tok(tokens, i+2) == "دارد"
            and _tok(tokens, i+3) == "می‌رود"
            and _tok(tokens, i-1) == "به"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "راستشون"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i-1) == "سمت"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "نمی‌بینم"
            and _tok(tokens, i-1) == "چیز"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i-1) == "که"
            and _tok(tokens, i+2) == "دارد"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "برگ"
            and _tok(tokens, i-1) == "از"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i-1) == "از"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) in {"یک‌دانه","یکدانه"}
            and _tok(tokens, i+1) == "روزنامه"
            and _tok(tokens, i-1) == "با"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبه‌روی","روبهروی"}
            and _tok(tokens, i+1) == "معدن"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "معدن"
            and _tok(tokens, i+1) == "پیک‌نیک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "وارد"
            and _tok(tokens, i+1) == "شود"
            and _tok(tokens, i-1) == "که"
            and _tok(tokens, i-2) == "کرده"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "وسطش"
            and _tok(tokens, i+1) == "باز"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "برگشتن"
            and _tok(tokens, i+1) == "به"
            and _tok(tokens, i+2) == "آمده"
        ):
            _set_pos(t,"VERB")
            continue

        if (
            _canon(_tok(tokens, i)) in {"بههرحال","به‌هرحال"}
            and _tok(tokens, i+1) == "سکوت"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"بههرحال","به‌هرحال"}
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "را"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"آنطرفش","آن‌طرفش"}
            and _tok(tokens, i+1) == "هم"
            and _tok(tokens, i+2) == "درختان"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"سفره‌ای","سفرهای"}
            and _tok(tokens, i+1) == "پهن"
            and _tok(tokens, i+2) == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خواهر"
            and _tok(tokens, i+1) == "برادر"
            and _tok(tokens, i+2) == "هستند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فرش"
            and _tok(tokens, i+1) == "پهن"
            and _tok(tokens, i+2) == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "چیز"
            and _tok(tokens, i+3) == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نفر"
            and _tok(tokens, i+1) == "جمع"
            and _tok(tokens, i+2) == "هستند"
            and _tok(tokens, i-1) == "سه"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "پایین"
            and _tok(tokens, i-1) == "اول"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "ندارد"
            and _tok(tokens, i-1) == "این"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"پنجتا","پنج‌تا"}
        ):
            _set_pos(t, "NUM")
            _set_lemma(t, "پنج")
            continue

        if (
            _canon(_tok(tokens, i)) == "حافظ"
            and _tok(tokens, i+1) == "می‌خواند"
        ):
            _set_pos(t,"PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کشن"
            and _tok(tokens, i+1) == "چپق"
            and _tok(tokens, i-1) == "حال"
            and _tok(tokens, i-2) == "در"
        ):
            _set_token(t,"کشیدن")
            _set_lemma(t,"کشیدن")
            continue

        if (
            _canon(_tok(tokens, i)) == "قرار"
            and _tok(tokens, i+1) == "ترجیح"
            and _tok(tokens, i+2) == "داده"
            and _tok(tokens, i-1) == "به"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "افراد"
            and _tok(tokens, i+1) == "قلاب"
            and _tok(tokens, i+2) == "می‌گیرد"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i-2) == "از"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i+1) == "حضور"
            and _tok(tokens, i+2) == "دارند"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i-2) == "در"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i+1) == "او"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i-1) == "ظاهرا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i+1) == "او"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i-1) == "آقا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ابتداش"
            and _tok(tokens, i+1) == "هستند"
            and _tok(tokens, i-1) == "در"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "میز"
            and _tok(tokens, i+1) == "گردی"
            and _tok(tokens, i-1) == "یک"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حالا"
            and _tok(tokens, i+1) == "پسربچه"
            and _tok(tokens, i-1) == "یک"
        ):
            _set_pos(t, "ADV")
            _unmark_ez(t)
            continue



        if (
            _canon(_tok(tokens, i)) == "جلوتر"
            and _tok(tokens, i+1) == "باز"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "مغازه"
            and _tok(tokens, i+4) == "دیگر"
            and _tok(tokens, i+5) == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"زیبایی","زیبای"}
            and _tok(tokens, i+1) == "داده"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i-2) == "به"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"زیبایی","زیبای"}
            and _tok(tokens, i+1) == "را"
            and _tok(tokens, i-1) == "نهایت"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مغازه"
            and _tok(tokens, i-2) == "باز"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "دیگر"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"بههرحل","به‌هرحال"}
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "ظرف‌ها"
            and _tok(tokens, i+3) == "را"
            and _tok(tokens, i+4) == "دارد"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"به‌هرحال","بههرحال"}
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "حین"
            and _tok(tokens, i+1) == "پسر"
            and _tok(tokens, i-1) == "همین"
            and _tok(tokens, i-2) == "در"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ورزشی"
            and _tok(tokens, i+1) == "پخش"
            and _tok(tokens, i+2) == "می‌کند"
            and _tok(tokens, i-1) == "برنامه"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"یک‌نوع","یکنوع"}
            and _tok(tokens, i+1) == "قایق"
            and _tok(tokens, i+2) == "می‌بیند"
            and _tok(tokens, i-1) == "اینجا"

        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "سیر"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "هستش"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کردی"
            and _tok(tokens, i-1) == "لباس"
        ):
            _set_pos(t, "ADJ")
            _set_lemma(t, "کردی")
            continue

        if (
            _canon(_tok(tokens, i)) == "وقت"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "مامان"
            and _tok(tokens, i-1) == "آن"

        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "بشقاب"
            and _tok(tokens, i+1) == "خشک"
            and _tok(tokens, i+2) == "می‌کند"

        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "قلعه"
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i+2) == "کرده"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "استاد"
            and _tok(tokens, i+1) == "غفاری"
            and _tok(tokens, i+2) == "دیگر"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقاپسر"
            and _tok(tokens, i+1) == "بالای"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "میز"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i+1) == "سرریز"
            and _tok(tokens, i+2) == "کرده"
            and _tok(tokens, i+3) == "پایین"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "شاخه‌ها"
            and _tok(tokens, i+2) == "دارد"
            and _tok(tokens, i+3) == "گیر"

        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i+2) == "تلویزیون"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "طوفانی"
            and _tok(tokens, i+1) == "می‌شود"
            and _tok(tokens, i-1) == "وقت"
            and _tok(tokens, i-2) == "یک"

        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سه‌پایه"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "میز"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "درخت"
        ):
            _set_pos(t, "ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "غفاری"
            and _tok(tokens, i-1) == "استاد"
            and _tok(tokens, i+1) == "دیگر"

        ):
            _unmark_ez(t)
            _set_pos(t, "PROPN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "غفاری"
            and _tok(tokens, i-2) == "استاد"
            and _tok(tokens, i+1) == "نهایت"

        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "دقیقا"
            and _tok(tokens, i+3) == "ماشین"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1) == "پنجره"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "منظره"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "آقا"
            and _tok(tokens, i+3) == "که"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) in {"مبل","مبل‌ها","تصویر","سرش"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "وسط"
            and _tok(tokens, i+1) == "بازار"
            and _tok(tokens, i-1) == "قسمت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "گونه‌اش"
            and _tok(tokens, i+2) == "گذاشته"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "آه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "لاله‌ها"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i-1) == "چارقد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"چانه‌اش","چانهاش"}
        ):
            _set_lemma(t,"چانه")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "شانه‌اش"
            and _tok(tokens, i-1) == "کوچولو"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i-1) == "را"
            and _tok(tokens, i-2) == "دستش"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "همدیگر"
            and _tok(tokens, i+2) == "هستند"

        ):
            _mark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-2) == "در"
            and _tok(tokens, i-1) == "کنار"

        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "سفره"
            and _tok(tokens, i-1) == "نشسته"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبه‌روی","روبهروی"}
            and _tok(tokens, i+1) == "ظرف‌شویی"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ظرف‌شویی","ظرفشویی"}
            and _tok(tokens, i-1) == "روبه‌روی"
            and _tok(tokens, i+1) == "پنجره"
        ):
            _unmark_ez(t)
            continue
            
            
            
        if (
            _canon(_tok(tokens, i)) == "لباس"
            and _tok(tokens, i+1) == "کردی"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ایران","کاشان","کمالالملک","کمال‌الملک"}
        ):
            _set_pos(t,"PROPN")
            continue


        if (
            _canon(_tok(tokens, i)) in {"طبقه‌ای","طبقهای"}
        ):
            _set_lemma(t,"طبقه")
            continue

        if (
            _canon(_tok(tokens, i)) == "مخیرم"
        ):
            _set_lemma(t,"مخیر")
            _set_pos(t, "ADJ")
            continue

        if _canon(_tok(tokens, i)) in {"اصفه‌اند","اصفهاند"}:
            _set_token(t, "اصفهان")
            _set_lemma(t, "اصفهان")
            _set_pos(t, "PROPN")
            t["morph_segments"] = []
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i-1) == "در"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"خانوادهاش","خانواده‌اش"}
        ):
            _set_lemma(t,"خانواده")
            continue

        if (
            _canon(_tok(tokens, i)) in {"مطالعه‌ام","مطالعهام"}
            and _tok(tokens, i-1) == "عینک"
        ):
            _set_pos(t,"NOUN")
            _set_lemma(t,"مطالعه")
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"مطالعه","role":"stem","morph_pos":"N"},
    {"form":"ام","role":"PRON_CL","morph_pos":"PRON","person":"1","number":"Sing","case":"Poss"},
]
            continue

        if (
            _canon(_tok(tokens, i)) == "والیبال"
            and _tok(tokens, i+1) == "درست"
            and _tok(tokens, i+2) == "می‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i-1) == "نمی‌فهمم"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "جمع"
            and _tok(tokens, i+1) == "هستند"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "دور"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یکی"
            and _tok(tokens, i+1) == "است"
            and _tok(tokens, i+2) == "یعنی"
            and _tok(tokens, i+3) == "سگه"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1) == "قابل"
            and _tok(tokens, i+2) == "توضیح"
            and _tok(tokens, i+3) == "نیست"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "دیگری"
            and _tok(tokens, i+3) == "نیست"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1) == "چیز"
            and _tok(tokens, i+2) == "دیگری"
            and _tok(tokens, i+3) == "را"
        ):
            _set_pos(t,"ADV")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "بعدی"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "پنجره"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "بالاتر"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانمی"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "دخترخانم"
            and _tok(tokens, i+3) == "نشسته"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "لیوانی"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "توپی"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پسر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "چیزی"
            and _tok(tokens, i+3) == "گذاشته"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "پرچم"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تاریخی"
            and _tok(tokens, i-1) == "این"
            and _tok(tokens, i+1) == "هست"
        ):
            _set_pos(t,"ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مدادی"
            and _tok(tokens, i-1) == "خودکاری"
            and _tok(tokens, i-2) == "یک"
        ):
            _set_pos(t,"NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1) == "هم"
            and _tok(tokens, i-2) == "این"
            and _tok(tokens, i+1) == "دوران"
            and _tok(tokens, i+2) == "ما"
            and _tok(tokens, i+3) == "را"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دوران"
            and _tok(tokens, i+1) == "ما"
            and _tok(tokens, i+2) == "را"
            and _tok(tokens, i+3) == "نشان"
            and _tok(tokens, i+4) == "می‌دهد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ظرف‌های","ظرفهای"}
            and _tok(tokens, i+1) == "قدیمی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "دوش"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "همان"
            and _tok(tokens, i+2) == "صندلی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "کتابخانه‌اش"
            and _tok(tokens, i+2) == "هست"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "چه"
            and _tok(tokens, i+2) == "هست"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) in {"دیوار","سفره","درخت","الاغ","میز"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1) == "زیر"
            and _tok(tokens, i+1) == "چانه‌اش"

        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"میوه‌خوری","میوهخوری"}
            and _tok(tokens, i-1) == "ظرف"
            and _tok(tokens, i+1) == "موز"
            and _tok(tokens, i+2) == "دارد"
        ):
            _set_pos(t,"NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قدیما"
            and _tok(tokens, i-1) == "دیگر"
            and _tok(tokens, i+1) == "بازار"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قالی"
            and _tok(tokens, i-1) == "که"
            and _tok(tokens, i-2) == "است"
            and _tok(tokens, i-3) == "چرخی"
            and _tok(tokens, i+1) == "درونش"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"سفره‌ای","سفرهای"}
            and _tok(tokens, i+1) == "انداختند"

        ):
            _unmark_ez(t)
            _set_lemma(t,"سفره")
            continue

        if (
            _canon(_tok(tokens, i)) in {"گربه‌ای","گربهای"}
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "آمده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i+1) == "چیزی"
            and _tok(tokens, i+2) == "دستش"
            and _tok(tokens, i+3) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) in {"یقه‌اش","یقهاش"}
        ):
            _set_lemma(t,"یقه")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i-2) == "در"
            and _tok(tokens, i+1) == "است"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i-1) == "روی"
            and _tok(tokens, i-2) == "گذاشته"
            and _tok(tokens, i-3) == "را"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1) == "یک"
            and _tok(tokens, i+1) == "،"
            and _tok(tokens, i+2) == "یک"
            and _tok(tokens, i+3) == "درخت"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "تلویزیون"
            and _tok(tokens, i+2) == "و"
            and _tok(tokens, i+3) == "یک"
            and _tok(tokens, i+4) == "یک‌دانه"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقای"
            and _tok(tokens, i+1) == "دارد"
            and _tok(tokens, i+2) == "کتاب"
            and _tok(tokens, i+3) == "می‌خواند"
        ):
            _set_token(t,"آقا")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "را"
            and _tok(tokens, i-1) == "آن"
            and _tok(tokens, i+1) == "بگیرد"
        ):
            _set_pos(t,"PART")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "سرش"
            and _tok(tokens, i+2) == "هم"
            and _tok(tokens, i+3) == "ساعت"
            and _tok(tokens, i+4) == "است"
        ):
            _set_pos(t,"ADP")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "درحقیقت"

        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "پنجره"
            and _tok(tokens, i-1) == "پایین"
            and _tok(tokens, i-2) == "از"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "پسر"
            and _tok(tokens, i+2) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مانند"
            and _tok(tokens, i-1) == "انباری"
            and _tok(tokens, i+1) == "با"
            and _tok(tokens, i+2) == "سقف‌های"
            and _tok(tokens, i+3) == "شیروانی"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌","آن","‌آن"}
        ):
            _set_token(t,"آن")
            _set_lemma(t,"آن")
            continue

        if (
            _canon(_tok(tokens, i)) == "معلوم"
            and _tok(tokens, i+1) == "نیست"
            and _tok(tokens, i-1) == "درست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1) == "درخت"
            and _tok(tokens, i-1) == "بروند"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i-1) == "حالا"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "حال"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+1) == "کشیدن"
            and _tok(tokens, i+2) == "سیگار"
            and _tok(tokens, i+3) == "هستند"
            
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حال"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+1) == "بافتنی"
            and _tok(tokens, i+2) == "بافتن"
            and _tok(tokens, i+3) == "هستند"
            
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حال"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+1) == "تلفن"
            and _tok(tokens, i+2) == "کردن"
            and _tok(tokens, i+3) == "است"
            
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بافتنی"
            and _tok(tokens, i+1) == "بافتن"
            and _tok(tokens, i+2) == "هستند"
            and _tok(tokens, i-1) == "حال"
            and _tok(tokens, i-2) == "در"
            
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "تلفن"
            and _tok(tokens, i+1) == "کردن"
            and _tok(tokens, i+2) == "است"
            and _tok(tokens, i-1) == "حال"
            and _tok(tokens, i-2) == "در"
            
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روزنامه‌ای","روزنامهای"}
            and _tok(tokens, i+1) == "دستش"
            and _tok(tokens, i+2) == "است" 
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فضای"
            and _tok(tokens, i+1) == "فضای"
            and _tok(tokens, i+2) == "باز"
            and _tok(tokens, i+3) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ترکی"
            and _tok(tokens, i+1) == "جوی"
            and _tok(tokens, i+2) == "می‌گوییم"
            and _tok(tokens, i-1) == "تو"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "آن"
            and _tok(tokens, i-1) == "در"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "روبه‌روی"
            and _tok(tokens, i+3) == "معدن"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "روبه‌روی"
            and _tok(tokens, i+3) == "ظرف‌شویی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1) == "جنگلی"
            and _tok(tokens, i+2) == "جایی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جنگلی"
            and _tok(tokens, i-1) == "درون"
            and _tok(tokens, i+1) == "جایی"
            and _tok(tokens, i+2) == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دختر"
            and _tok(tokens, i+1) == "ساعت"
            and _tok(tokens, i+2) == "هشت"
            and _tok(tokens, i+3) == "شب"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بادکنک"
            and _tok(tokens, i+1) == "هوا"
            and _tok(tokens, i+2) == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1) == "در"
            and _tok(tokens, i+1) == "کتاب"
            and _tok(tokens, i+2) == "است"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i+1) == "صحرایی"
            and _tok(tokens, i+2) == "جایی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "لای"
            and _tok(tokens, i+1) == "درخت"
            and _tok(tokens, i+2) == "گیر"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "قفسه‌ای"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "میزی"
            and _tok(tokens, i+3) == "که"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "یک"
            and _tok(tokens, i+2) == "میز"
            and _tok(tokens, i+3) == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بادبانی"
            and _tok(tokens, i+1) == "قایق"
            and _tok(tokens, i-1) == "است"
        ):
            _set_pos(t,"ADJ")
            _set_lemma(t,"بادبانی")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i+1) == "این"
            and _tok(tokens, i+2) == "درخت"
            and _tok(tokens, i+3) == "ایستادند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "تصویر"
            and _tok(tokens, i-1)  == "از"
        ):
            _set_pos(t, "DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "جمع"
            and _tok(tokens, i+1)  == "هستند"
            and _tok(tokens, i-1)  == "نفر"
            and _tok(tokens, i-2)  == "سه"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1)  == "به"
            and _tok(tokens, i-1)  == "حالا"
            and _tok(tokens, i+2)  == "چه"
            and _tok(tokens, i+3)  == "دردی"
            and _tok(tokens, i+4)  == "می‌خورد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1)  == "هست"
            and _tok(tokens, i-1)  == "حالا"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) in {"شیبدار","شیب‌دار"}
            and _tok(tokens, i-1)  == "،"
            and _tok(tokens, i-2)  == "دارد"
            and _tok(tokens, i-3)  == "سقف"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "وارد"
            and _tok(tokens, i+2)  == "می‌شود"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "همین"
            and _tok(tokens, i+2)  == "هست"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "این‌ها"
            and _tok(tokens, i+2)  == "کوچولو"
            and _tok(tokens, i+3)  == "هستند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "آن‌ها"
            and _tok(tokens, i+2)  == "ریزها"
            and _tok(tokens, i+3)  == "پیدا"
            and _tok(tokens, i+4)  == "نمی‌شوند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "والیبالیست"
            and _tok(tokens, i-1)  == "چون"
            and _tok(tokens, i+1)  == "نیستم"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "نیستم"
            and _tok(tokens, i-1)  == "والیبالیست"
            and _tok(tokens, i-2)  == "چون"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) in {"همدیگر","هم‌دیگر"}
            and _tok(tokens, i-1)  == "با"
            and _tok(tokens, i+1)  == "انداختن"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "نیستند"
            and _tok(tokens, i-1)  == "چیزی"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i-2)  == "بالای"
            and _tok(tokens, i-3)  == "اول"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "هرچه"
            and _tok(tokens, i+1)  == "به"
            and _tok(tokens, i+2)  == "ذهنتان"
            and _tok(tokens, i+3)  == "می‌آید"
            and _tok(tokens, i+4)  == "بازگو"
            and _tok(tokens, i+5)  == "کنید"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i+1)  == "لذت"
            and _tok(tokens, i+2)  == "می‌برند"
            and _tok(tokens, i-1)  == "از"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "توپ"
            and _tok(tokens, i+1)  == "پیدا"
            and _tok(tokens, i+2)  == "کردند"
            and _tok(tokens, i-1)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقاپسر"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "ظرف"
            and _tok(tokens, i-1)  == "این"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بیل"
            and _tok(tokens, i+1)  == "زدن"
            and _tok(tokens, i+2)  == "دیگر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زدن"
            and _tok(tokens, i-1)  == "بیل"
            and _tok(tokens, i+1)  == "دیگر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "'گیر'"
            and _tok(tokens, i-1)  == "درخت"
            and _tok(tokens, i+1)  == "می‌کند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "درخت"
            and _tok(tokens, i-1)  == "شاخه‌های"
            and _tok(tokens, i+1)  == "گیر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌طرفی","اینطرفی"}
            and _tok(tokens, i-1)  == "می‌رود"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i-1)  == "افتاده"
            and _tok(tokens, i+1)  == "اینجا"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1)  == "گرفته"
            and _tok(tokens, i-2)  == "را"
            and _tok(tokens, i-3)  == "کلوچه"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "چه"
            and _tok(tokens, i+2)  == "هست"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "،"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1)  == "زیر"
            and _tok(tokens, i+2)  == "یک"
            and _tok(tokens, i+3)  == "درخت"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدش"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "اینجا"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "سوار"
            and _tok(tokens, i+1)  == "نمی‌شوند"
            and _tok(tokens, i-1)  == "اصلا"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1)  == "بچه"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "که"
            and _tok(tokens, i+4)  == "دیده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شلوار"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i+2)  == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"بچهی","بچه‌ی"}
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "می‌آید"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بیلی"
            and _tok(tokens, i+1)  == "انجام"
            and _tok(tokens, i+2)  == "بدهد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "باز"
            and _tok(tokens, i+3)  == "کرده"
        ):
            _set_pos(t,"NOUN")


        if (
            _canon(_tok(tokens, i)) == "ساختمان"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  in {"چوبی","چوب"}
            and _tok(tokens, i+3)  == "هوا"
            and _tok(tokens, i+4)  == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"چوبی","چوب"}
            and _tok(tokens, i+1)  == "هوا"
            and _tok(tokens, i+2)  == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ساختمان"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "سره"
        ):
            _unmark_ez(t)
            continue
        
        if (
            _canon(_tok(tokens, i)) == "باشند"
            and _tok(tokens, i-1)  == "باید"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1)  == "صحبت"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _set_pos(t, "AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "دستش"
            and _tok(tokens, i+1)  == "کرده"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "حالا"
            and _tok(tokens, i+1)  == "می‌خواهد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "برق"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "وصل"
            and _tok(tokens, i+2)  == "نیست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تلویزیون"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "وصل"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کاجی"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "باشد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"دهندهی","دهنده‌ی"}
            and _tok(tokens, i-1)  == "نشان"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقای"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "شعر"
            and _tok(tokens, i+3)  == "حافظ"
            and _tok(tokens, i+4)  == "می‌خواند"
        ):
            _set_token(t,"آقا")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i-2)  == "پس"
            and _tok(tokens, i+1)  == "چه"
            and _tok(tokens, i+2)  == "هست"
        ):
            _set_token(t,"بالا")
            _set_pos(t,"ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تاشون"
            and _tok(tokens, i-1)  == "سه"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "همان"
            and _tok(tokens, i+2)  == "سگه"
            and _tok(tokens, i+3)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "قدیما"
            and _tok(tokens, i+1)  == "می‌گفتند"
            and _tok(tokens, i+2)  == "که"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i+1)  == "کرده"
            and _tok(tokens, i-1)  == "اینکه"
            and _tok(tokens, i-2)  == "مثل"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i-1)  == "تصویر"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "دستش"
            and _tok(tokens, i+1)  == "کرده"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i-1)  == "تصویر"
            and _tok(tokens, i+1)  == "هرچه"
            and _tok(tokens, i+2)  == "می‌بینید"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "مبارک"
            and _tok(tokens, i-1)  == "شما"
            and _tok(tokens, i-2)  == "عید"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"ماشاالله","آه‌آه","آهآه"}
        ):
            _set_pos(t,"INTJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "گاری"
            and _tok(tokens, i-1)  == "و"
            and _tok(tokens, i+1)  == "حمل"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانواده"
            and _tok(tokens, i-1)  == "بابای"
            and _tok(tokens, i+1)  == "سیگارش"
            and _tok(tokens, i+2)  == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "پریده"
            and _tok(tokens, i+2)  == "توپ"
            and _tok(tokens, i+3)  == "را"
            and _tok(tokens, i+4)  == "بگیرد"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "مثلا"
            and _tok(tokens, i+2)  == "وسیله"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "صافش"
            and _tok(tokens, i+3)  == "می‌کند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "چه"
            and _tok(tokens, i+2)  == "کار"
            and _tok(tokens, i+3)  == "دارد"
            and _tok(tokens, i+4)  == "می‌کند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "تعقیبش"
            and _tok(tokens, i+3)  == "کند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1)  == "که"
            and _tok(tokens, i-2)  == "هم"
            and _tok(tokens, i-3)  == "گربه"
            and _tok(tokens, i+1)  == "،"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "زیر"
            and _tok(tokens, i+3)  == "این"
            and _tok(tokens, i+4)  == "،"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1)  == "زیر"
            and _tok(tokens, i+1)  == "،"
            and _tok(tokens, i+2)  == "بچه‌ی"
            and _tok(tokens, i+3)  == "این"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1)  == "،"
            and _tok(tokens, i-2)  == "هم"
            and _tok(tokens, i-3)  == "اینجا"
            and _tok(tokens, i+1)  == "،"
            and _tok(tokens, i+2)  == "این"
            and _tok(tokens, i+3)  == "سیب"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1)  == "می‌خواهند"
            and _tok(tokens, i+1)  == "،"
            and _tok(tokens, i+2)  == "این"
            and _tok(tokens, i+3)  == "سیب"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1)  == "اینجا"
            and _tok(tokens, i+1)  == "،"
            and _tok(tokens, i+2)  == "آن"
            and _tok(tokens, i+3)  == "خانم"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1)  == "دارند"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "مواظب"
            and _tok(tokens, i+3)  == "هستند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1)  == "در"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "چیزی"
            and _tok(tokens, i+3)  == "می‌خواهند"
            and _tok(tokens, i+4)  == "بخورند"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "نگاه"
            and _tok(tokens, i+3)  == "می‌کنند"
        ):
            _set_pos(t, "NOUN")
            continue


        if (
            _canon(_tok(tokens, i)) == "باز"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "فرق"
            and _tok(tokens, i+3)  == "می‌کنند"
        ):
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "اینطوری"
            and _tok(tokens, i-1)  == "اینجا"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیرش"
            and _tok(tokens, i-1)  == "نشسته‌اند"
            and _tok(tokens, i-2)  == "این‌ها"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "زیرش"
            and _tok(tokens, i-1)  == "گذاشته"
            and _tok(tokens, i-2)  == "صندلی"
        ):
            _set_pos(t,"ADP")
            continue


        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i+1)  == "سوار"
            and _tok(tokens, i+2)  == "خر"
            and _tok(tokens, i+3)  == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آفتاب"
            and _tok(tokens, i+1)  == "نور"
            and _tok(tokens, i+2)  == "می‌تابد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "موقع"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "رسم"
            and _tok(tokens, i+2)  == "نبوده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i-1)  == "تصویر"
            and _tok(tokens, i+1)  == "هرچه"
            and _tok(tokens, i+2)  == "می‌بینید"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کاسه"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "وسط"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اصطلاح"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "فعالیتی"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اصطلاح"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i-2)  == "کارهای"
            and _tok(tokens, i+1)  == "مهمی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اصطلاح"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i-2)  == "از"
            and _tok(tokens, i+1)  == "منظره"
            and _tok(tokens, i+2)  == "هم"
            and _tok(tokens, i+3)  == "لذت"
            and _tok(tokens, i+4)  == "می‌برد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شهرستان"
            and _tok(tokens, i-1)  == "مثلا"
            and _tok(tokens, i+1)  == "من"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i-1)  == "آنجا"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i-1)  == "سیب"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i-1)  == "می‌رود"
            and _tok(tokens, i+1)  == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i-1)  == "همان‌جا"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i-1)  == "سیب"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            _set_pos(t, "NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i-1)  == "می‌رود"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i-1)  == "می‌رود"
            and _tok(tokens, i-2)  == "دارد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i+1)  == "نردبان"
            and _tok(tokens, i-1)  == "برویم"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i-1)  == "بروند"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i+2)  == "سیب"
            and _tok(tokens, i+3)  == "بچینند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دور"
            and _tok(tokens, i+1)  == "میزی"
            and _tok(tokens, i+2)  == "نشسته‌اند"
        ):
            _mark_ez(t)
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کی"
            and _tok(tokens, i+1)  == "است"
            and _tok(tokens, i-1)  == "نمی‌دانم"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "سر"
            and _tok(tokens, i-1)  == "بروند"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "طرف"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "کلبه"
            and _tok(tokens, i+3)  == "چوبی"
            and _tok(tokens, i+4)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "طرف"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "پرچم"
            and _tok(tokens, i+2)  == "هم"
            and _tok(tokens, i+3)  == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "یکی"
            and _tok(tokens, i+1)  == "تو"
            and _tok(tokens, i+2)  == "قایق"
            and _tok(tokens, i+3)  == "می‌خواهد"
            and _tok(tokens, i+4)  == "پارو"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "کوچه"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1)  == "که"
            and _tok(tokens, i-2)  == "کلبه‌هایی"
            and _tok(tokens, i+1)  == "ساحل"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "جلوی"
            and _tok(tokens, i+2)  == "الاغ"
            and _tok(tokens, i+3)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"پنجره‌ی","پنجرهی"}
            and _tok(tokens, i+1)  == "از"
            and _tok(tokens, i+2)  == "این‌ها"
            and _tok(tokens, i+3)  == "که"
            and _tok(tokens, i+4)  == "رنگ"
            and _tok(tokens, i+5)  == "می‌کنند"
        ):
            _set_token(t,"پنجره‌ای")
            _set_lemma(t, "پنجره")
            _set_pos(t,"NOUN")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بچه"
            and _tok(tokens, i-1)  == "مامان"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "بچه"
            and _tok(tokens, i+3)  == "کوچک‌تر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سال"
            and _tok(tokens, i-1)  == "که"
            and _tok(tokens, i-2)  == "بکنند"
            and _tok(tokens, i-3)  == "اعلام"
            and _tok(tokens, i+1)  == "تحویل"
            and _tok(tokens, i+2)  == "خواهد"
            and _tok(tokens, i+3)  == "شد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"یک‌دانه","یکدانه"}
            and _tok(tokens, i+1)  == "فیلم"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"NUM")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "لباس"
            and _tok(tokens, i+2)  == "محلی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درخت"
            and _tok(tokens, i-1)  == "بالای"
            and _tok(tokens, i+1)  == "ول"
            and _tok(tokens, i+2)  == "نمی‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "توپی"
            and _tok(tokens, i-1)  == "اگر"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "تو"
            and _tok(tokens, i+3)  == "درخت"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1)  == "شیر"
            and _tok(tokens, i+2)  == "اب"
            and _tok(tokens, i+3)  == "را"
            and _tok(tokens, i+4)  == "باز"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "آب"
            and _tok(tokens, i+1)  == "باز"
            and _tok(tokens, i+2)  == "گذاشته"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "توپ"
            and _tok(tokens, i+3)  == "دست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "سوار"
            and _tok(tokens, i+2)  == "چیز"
            and _tok(tokens, i+3)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "یکدانه"
            and _tok(tokens, i+1)  == "کلبه"
            and _tok(tokens, i+2)  == "مانندی"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "یکدانه"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i-2)  == "روی"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i+2)  == "هست"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "بدو"
            and _tok(tokens, i+1)  == "می‌دود"
            and _tok(tokens, i+2)  == "دنبال"
            and _tok(tokens, i+3)  == "گربه"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "بدو"
            and _tok(tokens, i+1)  == "می‌دود"
            and _tok(tokens, i+2)  == "دنبال"
            and _tok(tokens, i+3)  == "آن"
            and _tok(tokens, i+4)  == "گربه"
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "تاریخی"
            and _tok(tokens, i+2)  == "هست"
        ):
            _set_pos(t, "PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "چه"
            and _tok(tokens, i+2)  == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"آخ‌آخ‌آخ‌آخ","آخآخآخآخ"}
        ):
            _set_pos(t, "INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "طرف"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "فرار"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کوچولو"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "لذت"
            and _tok(tokens, i+2)  == "می‌برد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i-1)  == "هیچ"
            and _tok(tokens, i+1)  == "عصایش"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "گذاشته"
        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "درخت"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "نجات"
            and _tok(tokens, i+2)  == "بدهند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نفر"
            and _tok(tokens, i-1)  == "سه"
            and _tok(tokens, i-2)  == "این"
            and _tok(tokens, i+1)  == "نردبان"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "گرفتند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i-1)  == "آمده"
            and _tok(tokens, i+1)  == "هی"
            and _tok(tokens, i+2)  == "دارد"
            and _tok(tokens, i+3)  == "نگاه"
            and _tok(tokens, i+4)  == "می‌کند"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"دختر‌بچه","دختربچه"}
        ):
            _set_token(t, "دختربچه")
            continue

        if (
            _canon(_tok(tokens, i)) == "یکدانه"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i+2)  == "هست"
        ):
            _set_pos(t, "NUM")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالایش"
            and _tok(tokens, i-1)  == "تکه"
            and _tok(tokens, i+1)  == "پیدا"
            and _tok(tokens, i+2)  == "نیست"
        ):
            _set_pos(t, "NOUN")
            _set_lemma(t,"بالا")
            continue

        

        if (
            _canon(_tok(tokens, i)) == "نگاه‌ش"

        ):
            _set_token(t, "نگاهش")
            _set_lemma(t,"نگاه")
            _set_token(t," NOUN")
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"نگاه","role":"stem","morph_pos":"N"},
    {"form":"ش","role":"PRON_CL","morph_pos":"PRON","person":"3","number":"Sing","case":"Poss"},
]
            continue

        if (
            _canon(_tok(tokens, i)) == "شکمو"
            and _tok(tokens, i-1)  == "شیطون"
            and _tok(tokens, i-2)  == "پسر"
            and _tok(tokens, i+1)  == "رفته"
        ):
            _set_pos(t, "ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i-1)  == "آمده"
            and _tok(tokens, i-2)  == "سگه"
            and _tok(tokens, i+1)  == "دیگر"
        ):
            _set_pos(t, "ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنارش"
            and _tok(tokens, i-1)  == "زیرسیگاری"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کلبه"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "چوبی"
            and _tok(tokens, i+2)  == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ساختمان"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "کوچک"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مجسمه"
            and _tok(tokens, i+1)  == "مانند"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "طرف"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "کلبه"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "طرف"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "پرچم"
            and _tok(tokens, i+2)  == "هم"
            and _tok(tokens, i+3)  == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بادبادک"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "پسره"
            and _tok(tokens, i+1)  == "درست"
            and _tok(tokens, i+2)  == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1)  == "شیر"
            and _tok(tokens, i+2)  == "آب"
            and _tok(tokens, i+3)  == "را"
            and _tok(tokens, i+4)  == "باز"
            and _tok(tokens, i+5)  == "کرده"
            
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "قبول"
            and _tok(tokens, i+1)  == "هست"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "نور"
            and _tok(tokens, i+2)  == "می‌تاید"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i-2)  == "که"
        ):
            _set_token(t, "بالا")
            _unmark_ez(t)
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "ذره"
            and _tok(tokens, i+1)  == "پایین‌تر"
            and _tok(tokens, i-1)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "تکه"
            and _tok(tokens, i+1)  == "بالایش"
            and _tok(tokens, i+2)  == "پیدا"
            and _tok(tokens, i+3)  == "نیست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بافتنیاش"
            and _tok(tokens, i-1)  == "وسایل"
            and _tok(tokens, i+1)  == "نشسته"
        ):
            _set_token(t, "بافتنی‌اش")
            _set_lemma(t,"بافتنی")
            _set_pos(t," NOUN")
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"بافتنی","role":"stem","morph_pos":"N"},
    {"form":"اش","role":"PRON_CL","morph_pos":"PRON","person":"3","number":"Sing","case":"Poss"},
]
            continue

        if (
            _canon(_tok(tokens, i)) == "نتوانست"
        ):
            t["had_pron_clitic"] = False
            t["morph_segments"] = [
    {"form":"ن","role":"PREF_NEG"},
    {"form":"توانستن","role":"stem","morph_pos":"V"},
    {"person":"3","number":"Sing","case":"Poss"},
]
            continue

        if (
            _canon(_tok(tokens, i)) in {"میوه‌ی","میوهی"}
            and _tok(tokens, i+1)  == "خیلی"
            and _tok(tokens, i+2)  == "عالی"
            and _tok(tokens, i+3)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دختر"
            and _tok(tokens, i+1)  == "سرگرم"
            and _tok(tokens, i+2)  == "همان"
            and _tok(tokens, i+3)  == "بافتنی"
            and _tok(tokens, i+4)  == "خودش"
            and _tok(tokens, i+5)  == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "من"
            and _tok(tokens, i+2)  == "چه"
            and _tok(tokens, i+3)  == "بگوییم"
        ):
            _set_pos(t, "ADV")
            continue

        

        if (
            _canon(_tok(tokens, i)) in {"میوه‌ی","میوهی"}
            and _tok(tokens, i+1)  == "خیلی"
            and _tok(tokens, i+2)  == "عالی"
            and _tok(tokens, i+3)  == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روزانه‌اشون"

        ):
            _set_lemma(t, "روزانه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"لباسهای","لباس‌های"}
            and _tok(tokens, i+1)  == "این‌ها"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  in {"الاغش","پشتش"}
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "بروند"
            and _tok(tokens, i+1)  == "میوه‌ها"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "بچینند"
        ):
            _set_token(t,"بالا")
            _set_pos(t,"ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مانند"
            and _tok(tokens, i-1)  == "مجسمه"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"ADJ")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "طرف"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "کلبه"
            and _tok(tokens, i+3)  == "چوبی"
            and _tok(tokens, i+4)  == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1)  == "بگذارد"
            and _tok(tokens, i-2)  == "می‌خواهد"
            and _tok(tokens, i+1)  == "این"
        ):
            _mark_ez(t)
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کوچک"
            and _tok(tokens, i-1)  == "خیلی"
            and _tok(tokens, i-2)  == "گل"
            and _tok(tokens, i-3)  == "یک"
            and _tok(tokens, i+1)  == "درونش"
            and _tok(tokens, i+2)  == "قرار"
            and _tok(tokens, i+3)  == "می‌گیرد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "گل"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "خیلی"
            and _tok(tokens, i+2)  == "کوچک"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "برود"
            and _tok(tokens, i+1)  == "گربه"
            and _tok(tokens, i+2)  == "از"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نفر"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i+2)  == "می‌خواهد"
            and _tok(tokens, i+3)  == "بیاندازد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبه‌رو","روبهرو"}
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "با"
            and _tok(tokens, i+2)  == "می‌آیند"
        ):
            _unmark_ez(t)
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "گربه‌ای"
            and _tok(tokens, i-3)  == "یک"
            and _tok(tokens, i+1)  == "می‌کند"
        ):
            _unmark_ez(t)
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i-2)  == "چیزی"
            and _tok(tokens, i+1)  == "آویزان"
        ):
            _unmark_ez(t)
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i-1)  == "چیزی"
            and _tok(tokens, i+1)  == "شبیه"
        ):
            _unmark_ez(t)
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i-1)  == "است"
            and _tok(tokens, i-2)  == "چیز"
            and _tok(tokens, i+1)  == "کنار"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "نظر"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i-2)  == "این"
            and _tok(tokens, i-3)  == "چون"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "قایق"
            and _tok(tokens, i+3)  == "می‌آید"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پای"
            and _tok(tokens, i+1)  == "او"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "مس"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i-1)  == "نمی‌دانم"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1)  == "هست"
            and _tok(tokens, i-2)  == "هم"
            and _tok(tokens, i-3)  == "سگ"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "پسره"
        ):
            _mark_ez(t)
            continue



        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i+2)  == "ایستاده"
            and _tok(tokens, i-1)  == "یعنی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "گاری"
            and _tok(tokens, i+2)  == "گذاشته‌اند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "دوشش"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "دوشش"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "گاری"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "مس"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1)  == "الاغ"
        ):
            _mark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "جلوی"
            and _tok(tokens, i+2)  == "الاغ"
            and _tok(tokens, i+3)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "جلوی"
            and _tok(tokens, i+2)  == "الاغ"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "اسکله"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "مس"
            and _tok(tokens, i-1)  == "می‌کند"
            and _tok(tokens, i-2)  == "قلم‌کاری"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "مس"
            and _tok(tokens, i-1)  == "می‌کند"
            and _tok(tokens, i-2)  == "قلم‌کاری"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "مس"
            and _tok(tokens, i-1)  == "می‌کند"
            and _tok(tokens, i-2)  == "قلم‌کاری"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1)  == "پرده"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "سالن"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1)  == "پرده"
            and _tok(tokens, i-1)  == "و"
            and _tok(tokens, i-2)  == "پرده‌ای"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1)  == "پنجره"
            and _tok(tokens, i+2)  == "هم"
            and _tok(tokens, i+3)  == "درخت"
        ):
            _mark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "خانمی"
            and _tok(tokens, i+1)  == "سمت"
            and _tok(tokens, i+2)  == "راست"
            and _tok(tokens, i+3)  == "ایستاده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "یک"
            and _tok(tokens, i+3)  == "سگ"
            and _tok(tokens, i+4)  == "است"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1)  == "می‌کند"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "گربه‌ای"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i-1)  == "سمت"
            and _tok(tokens, i-2)  == "بعد"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخری"
            and _tok(tokens, i-1)  == "بعد"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "رفته"
            and _tok(tokens, i+3)  == "بالا"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخری"
            and _tok(tokens, i+1)  == "،"

        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخری"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "که"
            and _tok(tokens, i+3)  == "باز"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخری"
            and _tok(tokens, i-1)  == "بعد"
            and _tok(tokens, i+1)  == "سگه"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1)  == "چه"
            and _tok(tokens, i+2)  == "چیزی"
            and _tok(tokens, i+3)  == "ممکن"
            and _tok(tokens, i+4)  == "است"
            and _tok(tokens, i+5)  == "آنجا"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1)  == "مثل"
            and _tok(tokens, i+2)  == "کار"
            and _tok(tokens, i+3)  == "چیز"
            and _tok(tokens, i+4)  == "را"
            and _tok(tokens, i+5)  == "انجام"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i-1)  == "هفت‌سین"

        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i-1)  == "که"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "تکه"

        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) in {"روبهرو","روبه‌رو"}
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "با"
            and _tok(tokens, i+1)  == "می‌آیند"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"پایین‌تر","پایینتر"}
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "می‌آید"

        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "مس"
            and _tok(tokens, i+1)  == "انجام"
            and _tok(tokens, i+2)  == "می‌دهد"
            and _tok(tokens, i-1)  == "کار"
            and _tok(tokens, i-2)  == "درواقع"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i+1)  == "جا"
            and _tok(tokens, i+2)  == "انداختم"
            and _tok(tokens, i+3)  == "بگذارم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i+1)  == "بوده"
            and _tok(tokens, i-1)  == "طرف"
            and _tok(tokens, i-2)  == "در"
            and _tok(tokens, i-3)  == "کلوچه"
            and _tok(tokens, i-4)  == "این"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "رو"
            and _tok(tokens, i+1)  == "به"
            and _tok(tokens, i+2)  == "بیرون"
            and _tok(tokens, i+3)  == "باشد"
            and _tok(tokens, i-1)  == "باید"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "آخری"
        ):
            _set_lemma(t,"آخر")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "ساعت"
            and _tok(tokens, i-2)  == "یک"
            and _tok(tokens, i+1)  == "چیز"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالاتر"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "آقایی"
            and _tok(tokens, i-3)  == "یک"
            and _tok(tokens, i+1)  == "نشسته"
        ):
            _set_pos(t,"ADV")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "گربه‌ای"
            and _tok(tokens, i-3)  == "یک"
            and _tok(tokens, i+1)  == "می‌کند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i-1)  == "دارد"
            and _tok(tokens, i-2)  == "را"
            and _tok(tokens, i-3)  == "گربه‌ای"
            and _tok(tokens, i-4)  == "یک"
            and _tok(tokens, i+1)  == "می‌کند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "وقت"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "گربه"
            and _tok(tokens, i+3)  == "دارد"
            and _tok(tokens, i-1)  == "آن"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جهت"
            and _tok(tokens, i+1)  == "باد"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "نشان"
            and _tok(tokens, i+4)  == "می‌دهد"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پنجره"
            and _tok(tokens, i+1)  == "آنها"
            and _tok(tokens, i+2)  == "است"
            and _tok(tokens, i-1)  == "جلوی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "تصویر"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1)  == "او"
            and _tok(tokens, i-1)  == "با"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1)  == "او"
            and _tok(tokens, i+2)  == "پیاده"
            and _tok(tokens, i+3)  == "هست"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1)  == "سیگار"
            and _tok(tokens, i+2)  == "می‌کشد"
        ):
            _set_pos(t,"AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i-1)  == "چپ"
            and _tok(tokens, i-2)  == "سمت"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "من"
            and _tok(tokens, i+2)  == "چیز"
            and _tok(tokens, i+3)  == "دیگری"
            and _tok(tokens, i+4)  == "که"
            and _tok(tokens, i+5)  == "اینجا"
            and _tok(tokens, i+6)  == "بخواهم"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آنطور"
            and _tok(tokens, i+1)  == "کشیده"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "درخت"
            and _tok(tokens, i-3)  == "از"
            and _tok(tokens, i-4)  == "سایه‌ای"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1)  == "درواقع"
            and _tok(tokens, i+2)  == "عکس‌العملی"
            and _tok(tokens, i+3)  == "را"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1)  == "خوابیده"
            and _tok(tokens, i+2)  == "درواقع"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1)  == "درمقابل"
            and _tok(tokens, i+2)  == "چیزهای"
            and _tok(tokens, i+3)  == "بزرگ"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i+1)  == "است"
            and _tok(tokens, i-1)  == "نمی‌شوم"
            and _tok(tokens, i-2)  == "متوجه"
            and _tok(tokens, i-3)  == "را"
            and _tok(tokens, i-4)  == "این"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "توده‌ای"
            and _tok(tokens, i-1)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1)  == "الاغ"
            and _tok(tokens, i+2)  == "هست"
        ):
            _mark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "پرچم"
            and _tok(tokens, i+2)  == "هم"
            and _tok(tokens, i+3)  == "هست"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "آشپزخونه"
            and _tok(tokens, i+2)  == "است"
            and _tok(tokens, i+3)  == "دیگر"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1)  == "آشپزخونه"
            and _tok(tokens, i-2)  == "است"
            and _tok(tokens, i-3)  == "دیگر"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"عکسالعملی","عکس‌العملی"}
            and _tok(tokens, i+1)  == "نشان"
            and _tok(tokens, i+2)  == "نمی‌دهد"
            and _tok(tokens, i-1)  == "هیچ"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "ساختمان"
            and _tok(tokens, i+3)  == "است"
        ):
            _mark_ez(t)
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"مغازهها","مغازه‌ها","مغازه"}
        ):
            _set_lemma(t,"مغازه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"پایهدار","پایه‌دار","پایهدارها","پایه‌دارها"}
        ):
            _set_lemma(t,"پایه‌دار")
            continue

        if (
            _canon(_tok(tokens, i)) in {"لیوان‌ها","لیوانها"}
            and _tok(tokens, i+1)  == "سه‌تا"
            and _tok(tokens, i+2)  in {"هستش","هست‌ش","است"}
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ایده‌ای","ایدهای"}
        ):
            _set_lemma(t,"ایده")
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1)  == "بافتنی"
            and _tok(tokens, i+2)  == "می‌بافد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "تصویر"
            and _tok(tokens, i+2)  == "هم"
            and _tok(tokens, i+3)  == "یک"
            and _tok(tokens, i+4)  == "درخت"
            and _tok(tokens, i+5)  == "است"
        ):
            _set_pos(t, "ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "دیوار"
            and _tok(tokens, i+2)  == "هست"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "روزنامه‌ای"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"پایین‌تر","پایینتر"}
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i-1)  == "ذره"
            and _tok(tokens, i-2)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوی"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i-1)  == "است"
            and _tok(tokens, i-2)  == "چهارپایه"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "همه"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "گفتم"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بیرون"
            and _tok(tokens, i+1)  == "شره"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "عید"
            and _tok(tokens, i-1)  == "درون"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i+2)  == "نیست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "عید"
            and _tok(tokens, i-1)  == "درون"
            and _tok(tokens, i+1)  == "انار"
            and _tok(tokens, i+2)  == "و"
            and _tok(tokens, i+3)  == "گلابی"
            and _tok(tokens, i+4)  == "نیست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "فکر"
            and _tok(tokens, i+2)  == "می‌کنم"
            and _tok(tokens, i+3)  == "دریا"
            and _tok(tokens, i+4)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i-1)  == "سگه"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "چیزی"
            and _tok(tokens, i+3)  == "می‌رود"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"
            and _tok(tokens, i+1)  == "احتمال"
            and _tok(tokens, i+2)  == "زیاد"
            and _tok(tokens, i+3)  == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "درخت"
            and _tok(tokens, i+3)  == "را"
            and _tok(tokens, i+4)  == "نگه"
            and _tok(tokens, i+5)  == "داشته"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "کدومش"
            and _tok(tokens, i+1)  == "درست"
            and _tok(tokens, i+2)  == "است"
            and _tok(tokens, i-1)  == "نمی‌دانم"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "تنها"
            and _tok(tokens, i-1)  == "فقط"
            and _tok(tokens, i+1)  == "جایی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "فال"
            and _tok(tokens, i+2)  == "می‌گیرد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "باز"
            and _tok(tokens, i+3)  == "می‌کند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "هست"
            and _tok(tokens, i-1)  == "دوتا"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1)  == "روی"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i-1)  == "می‌رود"
            and _tok(tokens, i+1)  == "گربه"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگه"
            and _tok(tokens, i+1)  == "احتمال"
            and _tok(tokens, i+2)  == "زیاد"
            and _tok(tokens, i+3)  == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بکند"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "میوه"
        ):
            _set_lemma(t,"کندن")
            continue

        if (
            _canon(_tok(tokens, i)) == "ماشین"
            and _tok(tokens, i+1)  == "پارک"
            and _tok(tokens, i+2)  == "کرده‌اند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "ضربه"
            and _tok(tokens, i+3)  == "محکم"
            and _tok(tokens, i+4)  == "می‌زند"
        ):
            _set_pos (t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "گیر"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _set_token (t,"بالا")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"مس‌فروش","مسفروش"}
            and _tok(tokens, i-1)  == "یا"
            and _tok(tokens, i-2)  == "است"
            and _tok(tokens, i-3)  == "مسگر"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"NOUN")
            _set_lemma(t,"مس‌فروش")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانمی"
            and _tok(tokens, i+1)  == "لباس"
            and _tok(tokens, i+2)  == "محلی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i-2)  == "هم"
            and _tok(tokens, i+1)  == "بیرون"
            and _tok(tokens, i+2)  == "دیده"
            and _tok(tokens, i+3)  == "می‌شود"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بکند"
            and _tok(tokens, i-1)  == "سیب"
        ):
            _set_lemma(t,"کندن")
            continue

        if (
            _canon(_tok(tokens, i)) == "بکند"
            and _tok(tokens, i-1)  == "میوه"
        ):
            _set_lemma(t,"کندن")
            continue

        if (
            _canon(_tok(tokens, i)) == "بکند"
            and _tok(tokens, i-1)  == "."
            and _tok(tokens, i-2)  == "کند"
            and _tok(tokens, i-3)  == "چیز"
            and _tok(tokens, i-4)  == "سیب"
        ):
            _set_lemma(t,"کندن")
            continue

        if (
            _canon(_tok(tokens, i)) == "بکند"
            and _tok(tokens, i-1)  == "بالا"
            and _tok(tokens, i-2)  == "برود"
            and _tok(tokens, i-3)  == "نمی‌خواهند"
        ):
            _set_lemma(t,"کندن")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "فرش"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "الاغ"
            and _tok(tokens, i-1)  == "روی"
            and _tok(tokens, i+1)  == "سوار"
            and _tok(tokens, i+2)  == "شده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "به"
            and _tok(tokens, i+3)  == "آن"
            and _tok(tokens, i+4)  == "طرف"
            and _tok(tokens, i+5)  == "است"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "راست"
            and _tok(tokens, i-1)  == "دست"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "چیزهایی"
        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i+1)  == "باز"
            and _tok(tokens, i+2)  == "می‌شود"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیوار"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "دوتا"
            and _tok(tokens, i+2)  == "عکس"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "پایش"
            and _tok(tokens, i+2)  == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دوستان"
            and _tok(tokens, i+1)  == "جمع"
            and _tok(tokens, i+2)  == "هستند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "صورت"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i-2)  == "به"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "گربه‌ای"
            and _tok(tokens, i+3)  == "را"
            and _tok(tokens, i+4)  == "می‌بیند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نفر"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "قالیچه"
            and _tok(tokens, i+2)  == "می‌برد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "رو"
            and _tok(tokens, i+1)  == "به"
            and _tok(tokens, i+2)  == "یک"
            and _tok(tokens, i+3)  == "جایی"
            and _tok(tokens, i+4)  == "هست"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "را"
            and _tok(tokens, i-1)  == "این‌ها"
            and _tok(tokens, i+1)  == "درست"
        ):
            _set_pos(t,"PART")
            continue

        if (
            _canon(_tok(tokens, i)) == "را"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "بگیرد"
        ):
            _set_pos(t,"PART")
            continue

        if (
            _canon(_tok(tokens, i)) == "سایر"
            and _tok(tokens, i-1)  == "و"
            and _tok(tokens, i-2)  == "سبزه"
            and _tok(tokens, i+1)  == "ملحقاتش"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "میز"
            and _tok(tokens, i-1)  == "روی"
            and _tok(tokens, i+1)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "سوار"
            and _tok(tokens, i+2)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"شمعدون‌های","شمعدونهای"}
        ):
            _set_pos(t,"NOUN")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "ده"
            and _tok(tokens, i+1)  == "آمده"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i-2)  == "نمی‌دانم"
        ):
            _set_pos(t,"NOUN")
            _set_lemma(t,"ده")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارند"
            and _tok(tokens, i+1)  == "به"
            and _tok(tokens, i+2)  == "کار"
            and _tok(tokens, i+3)  == "و"
            and _tok(tokens, i+4)  == "کاسبیشون"
            and _tok(tokens, i+5)  == "می‌رسند"
        ):
            _set_pos(t,"AUX")
            continue



        if (
            _canon(_tok(tokens, i)) in {"دهی","ده‌ی"}
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i+1)  == "جایی"
            and _tok(tokens, i+2)  == "آمده"
        ):
            _unmark_ez(t)
            _set_lemma(t,"ده")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "عکس"
            and _tok(tokens, i-2)  == "دوتا"
            and _tok(tokens, i-3)  == "آن"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "ندارم"
        ):
            _unmark_ez(t)
            _set_token(t,"بالا")
            continue

        if (
            _canon(_tok(tokens, i)) == "رفتند"
            and _tok(tokens, i-1)  == "بالا"
            and _tok(tokens, i-2)  == "بیخودی"
            and _tok(tokens, i-3)  == "درخت"
            and _tok(tokens, i-4)  == "از"
            and _tok(tokens, i-5)  == "می‌کند"
            and _tok(tokens, i-6)  == "شروع"
            and _tok(tokens, i-7)  == "سگه"
        ):
            _unmark_ez(t)
            _set_token(t,"رفتن")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "سوار"
            and _tok(tokens, i+2)  == "الاغش"
            and _tok(tokens, i+3)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1)  == "جلوی"
            and _tok(tokens, i+1)  == "قرار"
            and _tok(tokens, i+2)  == "گرفته"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1)  == "جلوی"
            and _tok(tokens, i+1)  == "مغازه"
            and _tok(tokens, i+2)  == "ایستادند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "کامواش"
            and _tok(tokens, i+1)  == "انداخته"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1)  == "در"
            and _tok(tokens, i+1)  == "این‌ها"
            and _tok(tokens, i+2)  == "نشسته"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "زباله"
            and _tok(tokens, i-1)  == "سطل"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "سطل"
            and _tok(tokens, i+1)  == "زباله"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پارکینگ"
            and _tok(tokens, i-1)  == "تو"
            and _tok(tokens, i+1)  == "پارک"
            and _tok(tokens, i+2)  == "کردند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "مشغول"
            and _tok(tokens, i+2)  == "مطالعه"
            and _tok(tokens, i+3)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "توپ"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "بگیرد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بادبادک"
            and _tok(tokens, i+1)  == "هوا"
            and _tok(tokens, i+2)  == "کردن"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دادند"
            and _tok(tokens, i-1)  == "تکان"
            and _tok(tokens, i-2)  == "حال"
            and _tok(tokens, i-3)  == "در"
        ):
            _set_token(t,"دادن")
            _set_lemma(t,"دادن")
            t["morph_segments"] = []
            continue

        if (
            _canon(_tok(tokens, i)) == "گرفتند"
            and _tok(tokens, i+1)  == "توپ"
            and _tok(tokens, i-1)  == "حال"
            and _tok(tokens, i-2)  == "در"
        ):
            _set_token(t,"گرفتن")
            _set_lemma(t,"گرفتن")
            _mark_ez(t)
            t["morph_segments"] = []
            continue

        if (
            _canon(_tok(tokens, i)) == "تکان"
            and _tok(tokens, i-1)  == "حال"
            and _tok(tokens, i+1)  in {"دادن","دادند"}
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"NOUN")
            t["morph_segments"] = []
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i-1)  == "قایقی"
            and _tok(tokens, i+2)  == "آب"
            and _tok(tokens, i+3)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سگی"
            and _tok(tokens, i+1)  == "وارد"
            and _tok(tokens, i+2)  == "باغی"
            and _tok(tokens, i+3)  == "شده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "موز"
            and _tok(tokens, i+1)  == "درونش"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "اینجا"
            and _tok(tokens, i-2)  == "بگوییم"
            and _tok(tokens, i+1)  == "می‌بینم"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "درخت"
            and _tok(tokens, i-1)  == "زیر"
            and _tok(tokens, i+1)  == "نگه"
            and _tok(tokens, i+2)  == "داشتند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شن"
            and _tok(tokens, i-1)  == "با"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "مثلا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i+1)  == "سر"
            and _tok(tokens, i+2)  == "آن"
            and _tok(tokens, i+3)  == "آقا"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بسته"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "درونش"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"NOUN")
            _set_lemma(t,"بسته")
            continue

        if (
            _canon(_tok(tokens, i)) in {"الان‌ها","الانها"}
            and _tok(tokens, i+1)  == "سال"
            and _tok(tokens, i+2)  == "تحویل"
            and _tok(tokens, i+3)  == "شد"
        ):
            _set_pos(t,"ADV")
            _set_lemma(t,"الان")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "سال"
            and _tok(tokens, i-1)  == "الان‌ها"
            and _tok(tokens, i+1)  == "تحویل"
            and _tok(tokens, i+2)  == "شد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پارو"
            and _tok(tokens, i-1)  == "با"
            and _tok(tokens, i+1)  == "چیز"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1)  == "دنبال"
            and _tok(tokens, i+2)  == "یک"
            and _tok(tokens, i+3)  == "گربه‌ای"
            and _tok(tokens, i+4)  == "می‌کند"
        ):
            _unmark_ez(t)
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دنبال"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "گربه‌ای"
            and _tok(tokens, i+3)  == "می‌کند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "است"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i-2)  == "هم"
            and _tok(tokens, i-3)  == "بیل"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i+1)  == "زدند"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "پرده‌ها"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "تابلو"
            and _tok(tokens, i+1)  == "بالای"
            and _tok(tokens, i+2)  == "تلویزیون"
            and _tok(tokens, i+3)  == "هست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مثل"
            and _tok(tokens, i+1)  == "اینکه"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانه"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i-2)  == "تو"
            and _tok(tokens, i+1)  == "زندگی"
            and _tok(tokens, i+2)  == "می‌کنند"
        ):
            _unmark_ez(t)
            continue


        if (
            _canon(_tok(tokens, i)) in {"آن‌طرفش","آنطرفش"}
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "که"
            and _tok(tokens, i+3)  == "یک"
            and _tok(tokens, i+4)  == "تلویزیون"
            and _tok(tokens, i+5)  == "است"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالاترش"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "باز"
            and _tok(tokens, i+3)  == "پنجره"
            and _tok(tokens, i+4)  == "باید"
            and _tok(tokens, i+5)  == "باشد"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "بالاترش"
            and _tok(tokens, i+2)  == "هم"
            and _tok(tokens, i+3)  == "باز"
            and _tok(tokens, i+4)  == "پنجره"
            and _tok(tokens, i+5)  == "باید"
            and _tok(tokens, i+6)  == "باشد"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "راستش"
            and _tok(tokens, i-1)  == "دست"
            and _tok(tokens, i-2)  == "با"
            and _tok(tokens, i-3)  == "نه"
        ):
            _set_pos(t,"NOUN")
            _set_lemma(t,"راست")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "دولنگه"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"NOUN")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پیدا"
            and _tok(tokens, i+1)  == "کنند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایینش"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "خانم"
            and _tok(tokens, i-3)  == "یک"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "چهارپایه"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "یکخورده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "یکخورده"
            and _tok(tokens, i-1)  == "چهارپایه"
            and _tok(tokens, i-2)  == "این"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "دست"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "دارد"
            and _tok(tokens, i+2)  == "توپ‌بازی"
            and _tok(tokens, i+3)  == "می‌کند"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اصطلاح"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "حرکت"
            
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "بردارد"
        ):
            _set_token(t,"بالا")
            _set_pos(t,"ADP")
            _unmark_ez(t)
        elif _canon(_tok(tokens, i)) == "بالا" and _tok(tokens, i-1)  == "از" and _tok(tokens, i+1)  == "چیزی" and _tok(tokens, i+2)  == "را":
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i-1)  == "یکیشان"
            and _tok(tokens, i+1)  == "با"
            and _tok(tokens, i+2)  == "شن"
        ):
            _set_pos(t,"AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "تو"
            and _tok(tokens, i+2)  == "ردیف"
            and _tok(tokens, i+3)  == "بالای"
        ):
            _set_pos(t,"PART")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1)  == "کشیده"
            and _tok(tokens, i+2)  == "شد"
            and _tok(tokens, i+3)  == "کنار"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i-2)  == "پارو"
            and _tok(tokens, i-3)  == "با"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i-1)  == "می‌افتد"
            and _tok(tokens, i-2)  == "که"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "طوفانی"
            and _tok(tokens, i-1)  == "احتمالا"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "را"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "دارد"
            and _tok(tokens, i+1)  == "نشان"
            and _tok(tokens, i+1)  == "می‌دهد"
        ):
            _set_pos(t,"PART")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "ببینم"
            and _tok(tokens, i-2)  == "بگذار"
            and _tok(tokens, i+1)  == "شده"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بکنت"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "میوه"
            and _tok(tokens, i-3)  == "آن"
        ):
            _set_lemma(t, "کندن")
            continue

        if (
            _canon(_tok(tokens, i)) == "آب"
            and _tok(tokens, i+1)  == "منتظر"
            and _tok(tokens, i+2)  == "چیزی"
            and _tok(tokens, i+3)  == "است"
        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "طرف"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "گربه"
        ):
            _unmark_ez(t)
            continue
                
        if (
            _canon(_tok(tokens, i)) == "موز"
            and _tok(tokens, i+1)  == "میوه"
            and _tok(tokens, i+2)  == "و"
            and _tok(tokens, i+3)  == "این"
            and _tok(tokens, i+4)  == "چیزها"
        ):
            _unmark_ez(t)
            continue

                        
        if (
            _canon(_tok(tokens, i)) == "بغل"
            and _tok(tokens, i+1)  == "دست"
            and _tok(tokens, i+2)  == "این"
            and _tok(tokens, i+3)  == "خانم"
        ):
            _mark_ez(t)
            continue

                                
        if (
            _canon(_tok(tokens, i)) == "دست"
            and _tok(tokens, i-1)  == "بغل"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "خانم"
        ):
            _mark_ez(t)
            continue

                                        
        if (
            _canon(_tok(tokens, i)) == "آقاپسر"
            and _tok(tokens, i+1)  == "ظرف"
            and _tok(tokens, i+2)  == "کلوچه"
        ):
            _unmark_ez(t)
            continue

                                                
        if (
            _canon(_tok(tokens, i)) == "بازار"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i-2)  == "تو"
            and _tok(tokens, i+1)  == "وجود"
            and _tok(tokens, i+2)  == "دارد"
        ):
            _unmark_ez(t)
            continue

                                                        
        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i-1)  == "چرا"
            and _tok(tokens, i+1)  == "؟"
        ):
            _set_pos(t,"ADV")
            continue

                                                                
        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i-1)  == "باز"
            and _tok(tokens, i-2)  == "هم"
            and _tok(tokens, i-3)  == "آقا"
            and _tok(tokens, i+1)  == "به"
            and _tok(tokens, i+2)  == "نظر"
            and _tok(tokens, i+3)  == "می‌رسد"
        ):
            _set_pos(t,"ADV")
            continue

                                                                        
        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i-1)  == "باز"
            and _tok(tokens, i-2)  == "هم"
            and _tok(tokens, i-3)  == "آقا"
            and _tok(tokens, i+1)  == "به"
            and _tok(tokens, i+2)  == "نظر"
            and _tok(tokens, i+3)  == "می‌رسد"
        ):
            _set_pos(t,"ADV")
            continue

                                                                                
        if (
            _canon(_tok(tokens, i)) in {"یکوری","یک‌وری"}
            and _tok(tokens, i-1)  == "باز"
            and _tok(tokens, i+1)  == "کجش"
            and _tok(tokens, i+2)  == "کرد"
        ):
            _set_pos(t,"ADV")
            continue

                                                                                        
        if (
            _canon(_tok(tokens, i)) in {"یکوری","یک‌وری"}
            and _tok(tokens, i-1)  == "باز"
            and _tok(tokens, i+1)  == "کجش"
            and _tok(tokens, i+2)  == "کرد"
        ):
            _set_pos(t,"ADV")
            continue

                                                
        if (
            _canon(_tok(tokens, i)) == "پشت"
            and _tok(tokens, i-1)  == "قسمت"
            and _tok(tokens, i-2)  == "این"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  in {"آقاپسری","آقاپسر"}
        ):
            _mark_ez(t)
            continue

                                                        
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "پرنده‌ها"
            and _tok(tokens, i+3)  == "که"
            and _tok(tokens, i+4)  == "بگذریم"
        ):
            _set_token(t,"بالا")
            _unmark_ez(t)
            continue

                                                
        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1)  == "برمی‌دارد"
        ):
            _set_pos(t,"AUX")
            continue

        
                                
        if (
            _canon(_tok(tokens, i)) in {"نوشابه‌شان","نوشابهشان"}
        ):
            _set_lemma(t,"نوشابه")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "وارد"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "خانه"
            and _tok(tokens, i+3)  == "می‌شود"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1)  == "می‌کنند"
        ):
            _set_pos (t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "می‌رود"
            and _tok(tokens, i+1)  == "این"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانمی"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "کوزه"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "می‌رود"
            and _tok(tokens, i+1)  in {"نردبان","درخت"}
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "تو"
            and _tok(tokens, i-1)  == "معروف"
            and _tok(tokens, i-2)  == "قول"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "می‌گذارد"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) in {"این‌طرف‌تر","اینطرفتر"}
            and _tok(tokens, i-1)  == "که"
            and _tok(tokens, i-2)  == "است"
            and _tok(tokens, i-3)  == "نرگس"
            and _tok(tokens, i+1)  == "هست"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  in {"بچه‌هاشون","بچههاشون","بچه‌هایشان","بچههایشان",}
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بیرون"
            and _tok(tokens, i+1)  == "از"
            and _tok(tokens, i+2)  == "یک"
            and _tok(tokens, i+3)  == "پرده"
        ):
            _set_pos(t,"ADP")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "هرکی"
            and _tok(tokens, i+1)  == "دارد"
            and _tok(tokens, i+2)  == "قرآن"
            and _tok(tokens, i+3)  == "را"
            and _tok(tokens, i+4)  == "می‌خواند"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) in {"شیشه‌اش","شیشهاش"}
        ):
            _set_lemma(t,"شیشه")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالایش"
            and _tok(tokens, i-1)  == "بعد"
            and _tok(tokens, i+1)  == "هم"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "وسط"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i+1)  == "عکس"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i+1)  == "این‌ها"
            and _tok(tokens, i+2)  == "می‌آیند"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i+1)  == "سر"
            and _tok(tokens, i+2)  == "این‌ها"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دعا"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "پرداختن"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "عکسی"
            and _tok(tokens, i+1)  == "چیزی"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دریا"
            and _tok(tokens, i-1)  == "تو"
            and _tok(tokens, i+1)  == "پیشروی"
            and _tok(tokens, i+2)  == "بکند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "مغازه"
            and _tok(tokens, i-1)  == "مقابل"
            and _tok(tokens, i-2)  == "در"
            and _tok(tokens, i+1)  == "شخصی"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "می‌بینم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پدر"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "در"
            and _tok(tokens, i+4)  == "حیاط"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i-1)  == "دوتا"
            and _tok(tokens, i-2)  == "این"
            and _tok(tokens, i+1)  == "یکی"
            and _tok(tokens, i+2)  == "دست"
            and _tok(tokens, i+3)  == "می‌گیرد"
        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "آقایی"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i+2)  == "را"
        ):
            _unmark_ez(t)
            continue
                
        if (
            _canon(_tok(tokens, i)) == "بادکنک"
            and _tok(tokens, i-1)  == "با"
            and _tok(tokens, i+1)  == "بازی"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "داخل"
            and _tok(tokens, i+1)  == "هستیم"
            and _tok(tokens, i+2)  == "ما"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"نقشونگارهای","نقش‌ونگارهای"}
        ):
            _set_pos(t,"NOUN")
            t["had_pliral_suffix"] = True
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i+1)  in {"انداختند","انداختن"}
            and _tok(tokens, i+2)  == "نشستند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i+1)  == "به"
            and _tok(tokens, i+2)  == "اصطلاح"
            and _tok(tokens, i+3)  == "آن"
            and _tok(tokens, i+4)  == "ظرفشویی"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "خلاصه"
            and _tok(tokens, i+1)  == "،"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "چرخ‌دستی"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "گرفته"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خلاصه"
            and _tok(tokens, i+1)  == "هفت‌سین"
            and _tok(tokens, i+2)  == "چیده"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "قدیما"
            and _tok(tokens, i+2)  == "بازار"
            and _tok(tokens, i+3)  == "و"
            and _tok(tokens, i+4)  == "بازار"
            and _tok(tokens, i+5)  == "مسگرها"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "بقیه"
            and _tok(tokens, i+2)  == "هم"
            and _tok(tokens, i+3)  == "تو"
            and _tok(tokens, i+4)  == "بازار"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "حضور"
            and _tok(tokens, i+1)  == "شما"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پرداختن"
            and _tok(tokens, i-1)  == "دعا"
            and _tok(tokens, i-2)  == "به"
        ):
            _set_pos(t,"VERB")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "را"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i+1)  == "پیدا"
            and _tok(tokens, i+2)  == "کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زن"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i-2)  == "عکس"
            and _tok(tokens, i+1)  == "درونش"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "جلوتر"
            and _tok(tokens, i-1)  == "می‌آید"
            and _tok(tokens, i-2)  == "کمی"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "چیزی"
            and _tok(tokens, i+3)  == "هم"
            and _tok(tokens, i+4)  == "آن"
            and _tok(tokens, i+5)  == "گوشه"
            and _tok(tokens, i+6)  == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1)  == "سرش"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "دست"
            and _tok(tokens, i+2)  == "بچه"
            and _tok(tokens, i+3)  == "هم"
            and _tok(tokens, i+4)  == "یک"
            and _tok(tokens, i+5)  == "چیزی"
            and _tok(tokens, i+6)  == "هست"
            
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "قسمت"
            and _tok(tokens, i+1)  == "بیرون"
            and _tok(tokens, i+2)  == "شیشه‌ها"
            and _tok(tokens, i+3)  == "شفاف"
            and _tok(tokens, i+4)  == "است" 
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بیرون"
            and _tok(tokens, i-1)  == "قسمت"
            and _tok(tokens, i+1)  == "شیشه‌ها"
            and _tok(tokens, i+2)  == "شفاف"
            and _tok(tokens, i+3)  == "است" 
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "ضمنا"
            and _tok(tokens, i-1)  == "و"
            and _tok(tokens, i+1)  == "هم" 
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1)  == "بازار"
            and _tok(tokens, i+2)  == "قدیمی" 
            and _tok(tokens, i+3)  == "است"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "یعنی"
            and _tok(tokens, i+1)  == "والیبال"
            and _tok(tokens, i+2)  == "بازی" 
            and _tok(tokens, i+3)  == "می‌کنند"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "پارکی"
            and _tok(tokens, i-1)  == "درون"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "دارد"
            and _tok(tokens, i+3)  == "راه"
            and _tok(tokens, i+4)  == "می‌رود"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدی"
            and _tok(tokens, i-1)  == "تصویر"
            and _tok(tokens, i-2)  == "تو"
            and _tok(tokens, i+1)  == "سگه"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "اصطلاح"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "باز"
            and _tok(tokens, i+2)  == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "فرش"
            and _tok(tokens, i+1)  == "انداختند"
            and _tok(tokens, i+2)  == "نشسته‌اند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شنی"
            and _tok(tokens, i-1)  == "مجسمه"
            and _tok(tokens, i-2)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "شنی"
            and _tok(tokens, i-1)  == "مجسمه"
            and _tok(tokens, i-2)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بزرگ"
            and _tok(tokens, i-1)  == "توپ"
            and _tok(tokens, i-2)  == "یک"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کلاس"
            and _tok(tokens, i+1)  == "اول"
            and _tok(tokens, i+2)  == "ابتدایی"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "ببینیم"
            and _tok(tokens, i+1)  == "شده"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "و"
            and _tok(tokens, i+1)  == "می‌شود"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "رفت"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "بالایش"
            and _tok(tokens, i+1)  == "هست"
        ):
            _set_pos(t,"NOUN")
            _set_lemma(t,"رف")
            continue

        if (
            _canon(_tok(tokens, i)) == "چپی"
            and _tok(tokens, i-1)  == "دست"
            and _tok(tokens, i-2)  == "تصویر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "خانم"
            and _tok(tokens, i+1)  == "خواب"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پرتی"
            and _tok(tokens, i+1)  == "روی"
            and _tok(tokens, i+2)  == "میز"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "پیرمرده"
            and _tok(tokens, i-1)  == "آقای"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن‌طرف‌ها","آنطرفها"}
            and _tok(tokens, i+1)  == "دیگر"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"ماهی‌ای","ماهیای"}
        ):
            _set_lemma(t,"ماهی")
            continue

        if (
            _canon(_tok(tokens, i)) == "سگ"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "مقداری"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "واق"
            and _tok(tokens, i+1)  == "وق"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "مثلا"
            and _tok(tokens, i+1)  == "می‌کند"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "مانده"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "آنجا"
            and _tok(tokens, i+2)  == "ایستاده"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "وسط"
            and _tok(tokens, i+2)  == "عکس"
        ):
            _set_pos(t,"DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعد"
            and _tok(tokens, i-1)  == "و"
            and _tok(tokens, i+1)  == "نمی‌رسد"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "تصویر"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "هم"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بالا"
            and _tok(tokens, i-1)  == "سطح"
            and _tok(tokens, i-2)  == "آن"
            and _tok(tokens, i-3)  == "به"
        ):
            _set_pos(t,"ADJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "سقف"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "نمی‌شوم"
            and _tok(tokens, i-2)  == "متوجه"
            and _tok(tokens, i+1)  == "می‌کنند"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "اطراف"
            and _tok(tokens, i-1)  == "بعد"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "که"
            and _tok(tokens, i+3)  == "فکر"
            and _tok(tokens, i+4)  == "کنم"
            and _tok(tokens, i+5)  == "درخت"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "ورودی"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "نفر"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "داخلش"
            and _tok(tokens, i+2)  == "دارد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "پرده"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "پنکه"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "درخت"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "پرچم"
        ):
            _set_pos(t,"PRON")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "بغل"
        ):
            _set_pos(t,"PRON")
            continue

                
        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "الاغ"
        ):
            _set_pos(t,"PRON")
            continue

                        
        if (
            _canon(_tok(tokens, i)) == "بیرون"
            and _tok(tokens, i-1)  == "آمدند"
            and _tok(tokens, i+1)  == "خانه"
            and _tok(tokens, i+2)  == "نشسته‌اند"
        ):
            _mark_ez(t)
            continue

                                
        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i-1)  == "آن"
            and _tok(tokens, i-2)  == "از"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "چیز"
        ):
            _mark_ez(t)
            continue

                                        
        if (
            _canon(_tok(tokens, i)) == "درون"
            and _tok(tokens, i-1)  == "که"
            and _tok(tokens, i-2)  == "چیزی"
            and _tok(tokens, i-3)  == "آن"
            and _tok(tokens, i+1)  == "دسته"
        ):
            _mark_ez(t)
            continue

                                                
        if (
            _canon(_tok(tokens, i)) == "دارد"
            and _tok(tokens, i+1)  == "با"
            and _tok(tokens, i+2)  == "حالا"
            and _tok(tokens, i+3)  == "پسرش"
            and _tok(tokens, i+4)  == "یا"
            and _tok(tokens, i+5)  == "دخترش"
            and _tok(tokens, i+6)  == "دارند"
            and _tok(tokens, i+7)  == "بازی"
            and _tok(tokens, i+8)  == "می‌کنند"
        ):
            _set_pos(t,"AUX")
            continue

                                                        
        if (
            _canon(_tok(tokens, i)) == "یکدانه"
            and _tok(tokens, i-1)  == "اینجا"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "چیز"
            and _tok(tokens, i+3)  == "بچه"
            and _tok(tokens, i+4)  == "است"
        ):
            _set_pos(t,"NUM")
            continue
                                             
        if (
            _canon(_tok(tokens, i)) == "ای"
        ):
            _set_pos(t,"INTJ")
            continue

                                                                
        if (
            _canon(_tok(tokens, i)) in {"ساختمون‌های","ساختمونهای"}
            and _tok(tokens, i-1)  == "که"
            and _tok(tokens, i+1)  == "آنجا"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

                        
        if (
            _canon(_tok(tokens, i)) in {"همون‌جوری","همونجوری"}
            and _tok(tokens, i+1)  == "که"
            and _tok(tokens, i+2)  == "گفتم"
        ):
            _set_pos(t,"ADV")
            continue

                
        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "باز"
            and _tok(tokens, i+3)  == "سگه"
        ):
            _set_pos(t,"PRON")
            continue

                        
        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i-1)  == "داستانش"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

                                
        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "می‌گوییم"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

                                        
        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "شکلش"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "نشان"
            and _tok(tokens, i+3)  == "می‌دهد"
        ):
            _set_pos(t,"PRON")
            continue

        
                                        
        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "همه"
            and _tok(tokens, i+1)  == "در"
            and _tok(tokens, i+2)  == "آن"
            and _tok(tokens, i+3)  == "هست"
        ):
            _set_pos(t,"PRON")
            continue

                                                
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "رفته"
            and _tok(tokens, i+1)  == "چیز"

        ):
            _set_pos(t,"ADP")
            continue

                                                        
        if (
            _canon(_tok(tokens, i)) == "افطار"
            and _tok(tokens, i-1)  == "برای"
            and _tok(tokens, i+1)  == "قرآن"
            and _tok(tokens, i+2)  == "می‌خواند"

        ):
            _unmark_ez(t)
            continue

                                                                
        if (
            _canon(_tok(tokens, i)) == "میز"
            and _tok(tokens, i-1)  == "روی"
            and _tok(tokens, i+1)  == "وسط"
            and _tok(tokens, i+2)  in {"است","هست"}

        ):
            _mark_ez(t)
            continue

                                                                        
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "گل"
            and _tok(tokens, i-2)  == "دسته"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "کتابخونه"

        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        
                                                                        
        if (
            _canon(_tok(tokens, i)) == "گل"
            and _tok(tokens, i-1)  == "دسته"
            and _tok(tokens, i+1)  == "بالای"
            and _tok(tokens, i+2)  == "آن"
            and _tok(tokens, i+3)  == "کتابخونه"

        ):
            _unmark_ez(t)
            continue

                                                                                
        if (
            _canon(_tok(tokens, i)) == "خب"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "تصویر"
            and _tok(tokens, i-3)  == "این"
            and _tok(tokens, i+1)  == "خانم"

        ):
            _set_pos(t,"INTJ")
            continue

                                                                                        
        if (
            _canon(_tok(tokens, i)) == "خب"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "بابا"
            and _tok(tokens, i+3)  == "دارد"

        ):
            _set_pos(t,"INTJ")
            continue

                                                                                                
        if (
            _canon(_tok(tokens, i)) == "وسط"
            and _tok(tokens, i-1)  == "میز"
            and _tok(tokens, i-2)  == "روی"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"NOUN")
            continue

                                                                                
        if (
            _canon(_tok(tokens, i)) == "بایست"
            and _tok(tokens, i-1)  == "گهواره"
            and _tok(tokens, i+1)  == "باشد"

        ):
            _set_pos(t,"AUX")
            continue

                                                                                        
        if (
            _canon(_tok(tokens, i)) == "بایست"
            and _tok(tokens, i-1)  == "اینجا"
            and _tok(tokens, i-2)  == "هم"
            and _tok(tokens, i-3)  == "دریاچه‌ای"
            and _tok(tokens, i+1)  == "باشد"

        ):
            _set_pos(t,"AUX")
            continue

                                                
        if (
            _canon(_tok(tokens, i)) == "فردوسی"
        ):
            _set_pos(t,"PROPN")
            continue


        if (
            _canon(_tok(tokens, i)) == "اوهوم"
        ):
            _set_pos(t,"INTJ")
            continue


        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "سگ"
            and _tok(tokens, i+3)  == "دارد"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "یکی"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "پرنده‌ای"
            and _tok(tokens, i+3)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i-1)  == "ابنجا"
            and _tok(tokens, i-2)  == "بعد"
            and _tok(tokens, i+1)  == "را"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "خانم"
            and _tok(tokens, i+1)  == "دستش"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "یک"
            and _tok(tokens, i+1)  == "دستش"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i+1)  == "کاری"
            and _tok(tokens, i+2)  == "بکنیم"
        ):
            _set_pos(t,"PRON")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "کی"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"PRON")
            continue


        if (
            _canon(_tok(tokens, i)) in {"همراه‌اش","همراهاش"}
        ):
            _set_token(t,"همراهش")
            _set_lemma(t,"همراه")
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"همراه","role":"stem","morph_pos":"N"},
    {"form":"ش","role":"PRON_CL","morph_pos":"PRON","person":"3","number":"Sing","case":"Poss"},
]
            continue

        
        if (
            _canon(_tok(tokens, i)) == "خودش"
        ):
            _set_token(t,"خودش")
            _set_lemma(t,"خود")
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"خود","role":"stem","morph_pos":"N"},
    {"form":"ش","role":"PRON_CL","morph_pos":"PRON","person":"3","number":"Sing","case":"Poss"},
]
            continue

        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i-1)  == "او"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i-1)  == "شبیه"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "چه"
            and _tok(tokens, i-1)  == "ما"
            and _tok(tokens, i-2)  == "این‌جا"
            and _tok(tokens, i+1)  == "داریم"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "را"
            and _tok(tokens, i-1)  == "این‌ها"
            and _tok(tokens, i+1)  == "بگیرد"
        ):
            _set_pos(t,"PART")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "را"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "دارد"
            and _tok(tokens, i+2)  == "به"
            and _tok(tokens, i+3)  == "ما"
            and _tok(tokens, i+4)  == "نشان"
            and _tok(tokens, i+5)  == "می‌دهد"
        ):
            _set_pos(t,"PART")
            continue

        if (
            _canon(_tok(tokens, i)) in {"پنکه‌مان","پنکهمان"}
        ):
            _set_lemma(t,"پنکه")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "یک"
            and _tok(tokens, i+3)  == "میز"
        ):
            _set_pos(t,"PRON")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "کل"
            and _tok(tokens, i+3)  == "این"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "والیبال"
            and _tok(tokens, i+1)  == "بازی"
            and _tok(tokens, i+2)  == "می‌کنند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "صندلی"
            and _tok(tokens, i-1)  == "همان"
            and _tok(tokens, i+1)  == "تکیه"
            and _tok(tokens, i+2)  == "کرده"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "موز"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "آقا"
            and _tok(tokens, i-1)  == "این"
            and _tok(tokens, i+1)  == "صندلی"
            and _tok(tokens, i+2)  == "جدا"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "چیزی"
            and _tok(tokens, i-1)  == "عکسی"
            and _tok(tokens, i+1)  == "اینطوری"
        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i+1)  == "فرار"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _set_pos(t,"ADV")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "عتیقه"
            and _tok(tokens, i+1)  == "فروشی"
            and _tok(tokens, i+2)  == "است"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "کنار"
            and _tok(tokens, i-1)  == "اینجا"
            and _tok(tokens, i+1)  == "زن"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "فروشی"
            and _tok(tokens, i-1)  == "عتیقه"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "من"
            and _tok(tokens, i+3)  == "نمی‌بینم"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "دیگر"
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "ندارد"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "بعدش"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "کل"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "این"
            and _tok(tokens, i+1)  == "این"
            and _tok(tokens, i+2)  == "دارد"
            and _tok(tokens, i+3)  == "دعا"
            and _tok(tokens, i+4)  == "می‌کند"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بغل"
            and _tok(tokens, i-1)  == "می‌گذارند"
            and _tok(tokens, i+1)  == "درخت"
        ):
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "نوشابه"
            and _tok(tokens, i-1)  == "دارد"
            and _tok(tokens, i+1)  == "باز"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "عقب"
            and _tok(tokens, i-1)  == "سمت"
            and _tok(tokens, i-2)  == "به"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "داخل"
            and _tok(tokens, i-1)  == "ماشین"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "هستیم"
            and _tok(tokens, i-1)  == "داخل"
            and _tok(tokens, i+1)  == "ما"
        ):
            _set_pos(t,"AUX")
            continue

        if (
            _canon(_tok(tokens, i)) == "ظاهر"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "موز"
            and _tok(tokens, i+2)  == "می‌رسد"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"میوه‌اش","میوهاش"}
        ):
            _set_lemma(t,"میوه")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i-2)  == "دارد"
            and _tok(tokens, i+1)  == "آقا"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) in {"گربه","‌گربه","گربه‌"}
            and _tok(tokens, i+1)  == "خواب"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_lemma(t,"گربه")
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "باز"
            and _tok(tokens, i+3)  == "کردند"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "آن"
            and _tok(tokens, i+3)  == "باز"
            and _tok(tokens, i+4)  == "کرده"
        ):
            _set_pos(t,"NOUN")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "داخل"
            and _tok(tokens, i-1)  == "می‌شود"
            and _tok(tokens, i-2)  == "وارد"
            and _tok(tokens, i-3)  == "در"
            and _tok(tokens, i-4)  == "از"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "بادبادک"

        ):
            _set_pos(t,"DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i-1)  == "به"
            and _tok(tokens, i+1)  == "طرف"

        ):
            _set_pos(t,"DET")
            continue

        
        if (
            _canon(_tok(tokens, i)) == "آن"
            and _tok(tokens, i+1)  == "کتاب"
            and _tok(tokens, i+2)  == "فردوسی"

        ):
            _set_pos(t,"DET")
            continue

        if (
            _canon(_tok(tokens, i)) == "داخل"
            and _tok(tokens, i+1)  == "هستیم"
            and _tok(tokens, i+2)  == "ما"

        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) in {"آتش‌سوزی‌ای","آتشسوزیای"}
            and _tok(tokens, i+1)  == "چیزی"
            and _tok(tokens, i+2)  == "داخل"

        ):
            _unmark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "کلبه"
            and _tok(tokens, i-1)  == "یکسری"
            and _tok(tokens, i+1)  == "در"
            and _tok(tokens, i+2)  == "نظر"
            and _tok(tokens, i+3)  == "بگیریم"

        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "زیر"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i+1)  == "می‌خورد"
            and _tok(tokens, i+2)  == "به"
            and _tok(tokens, i+3)  == "این"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "درست"
            and _tok(tokens, i-1)  == "والیبال"
            and _tok(tokens, i+1)  == "بازی"
            and _tok(tokens, i+2)  == "می‌کنند"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "خلاصه"
            and _tok(tokens, i-1)  == "را"
            and _tok(tokens, i-2)  == "آن"
            and _tok(tokens, i+1)  == "از"
            and _tok(tokens, i+2)  == "آنجا"
        ):
            _set_pos(t,"INTJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "تا"
            and _tok(tokens, i-1)  == "هم"
            and _tok(tokens, i-2)  == "بالا"
            and _tok(tokens, i-3)  == "رفتند"
            and _tok(tokens, i+1)  == "برسند"
            and _tok(tokens, i+2)  == "به"
            and _tok(tokens, i+3)  == "آن"
        ):
            _set_pos(t,"SCONJ")
            continue

        if (
            _canon(_tok(tokens, i)) == "روی"
            and _tok(tokens, i-1)  == "از"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "کتاب"
            and _tok(tokens, i+3)  == "فردوسی"
        ):
            _set_pos(t,"ADP")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "بیرون"
            and _tok(tokens, i+1)  == "از"
            and _tok(tokens, i+2)  == "خانه"
        ):
            _set_pos(t,"ADP")
            continue

        if (
            _canon(_tok(tokens, i)) == "چی"
            and _tok(tokens, i-1)  == "شبیه"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "دهنده"
            and _tok(tokens, i-1)  == "نشان"
            and _tok(tokens, i+1)  == "آن"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"ADJ")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "دهنده"
            and _tok(tokens, i-1)  == "نشان"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "چیز"
        ):
            _set_pos(t,"ADJ")
            _mark_ez(t)
            continue

        
        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i-1)  == "می‌کند"
            and _tok(tokens, i-2)  == "تابش"
            and _tok(tokens, i+1)  == "مشخص"
            and _tok(tokens, i+2)  == "است"
        ):
            _set_pos(t,"ADV")
            continue

                
        if (
            _canon(_tok(tokens, i)) == "گذشته"
            and _tok(tokens, i-1)  == "دوران"
            and _tok(tokens, i-2)  == "به"
            and _tok(tokens, i-3)  == "باز"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"ADJ")
            continue

                        
        if (
            _canon(_tok(tokens, i)) == "بالای"
            and _tok(tokens, i-1)  == "دارد"
            and _tok(tokens, i+1)  == "یک"
            and _tok(tokens, i+2)  == "چیزی"
            and _tok(tokens, i+3)  == "می‌گیرد"
        ):
            _set_pos(t,"ADP")
            continue

                                
        if (
            _canon(_tok(tokens, i)) == "عینک"
            and _tok(tokens, i+1)  == "مطالعه‌ام"
        ):
            _mark_ez(t)
            continue

                                        
        if (
            _canon(_tok(tokens, i)) == "مشغول"
            and _tok(tokens, i+1)  == "خوردن"
            and _tok(tokens, i+2)  == "است"
        ):
            _mark_ez(t)
            continue

                                                
        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "او"
            and _tok(tokens, i+3)  == "باز"
            and _tok(tokens, i+4)  == "کرده"
        ):
            _set_pos(t,"NOUN")
            continue

                                                        
        if (
            _canon(_tok(tokens, i)) == "بغل"
            and _tok(tokens, i+1)  == "است"
            and _tok(tokens, i-1)  == "که"
            and _tok(tokens, i-2)  == "این"
        ):
            _set_pos(t,"NOUN")
            continue

                                                                
        if (
            _canon(_tok(tokens, i)) == "باعث"
            and _tok(tokens, i+1)  == "می‌شود"
        ):
            _set_pos(t,"NOUN")
            continue

                                                        
        if (
            _canon(_tok(tokens, i)) in {"پنجره‌مان","پنجرهمان"}
        ):
            _set_lemma(t,"پنجره")
            continue

                                                                
        if (
            _canon(_tok(tokens, i)) in {"پرده‌مان","پردهمان"}
        ):
            _set_lemma(t,"پرده")
            continue

        if (
            _canon(_tok(tokens, i)) == "افطار"
            and _tok(tokens, i-1)  == "برای"
            and _tok(tokens, i-2)  == "دارد"
            and _tok(tokens, i+1)  == "قرآن"
            and _tok(tokens, i+2)  == "می‌خواند"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "پایین"
            and _tok(tokens, i-1)  == "می‌پرد"
            and _tok(tokens, i-2)  == "گربه"
        ):
            _set_pos(t,"ADV")
            continue

        if (
            _canon(_tok(tokens, i)) == "این"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "عتیقه"
            and _tok(tokens, i+3)  == "فروشی"
            and _tok(tokens, i+4)  == "است"
        ):
            _set_pos(t,"PRON")
            continue

        if (
            _canon(_tok(tokens, i)) == "آمدند"
            and _tok(tokens, i-1)  == "بالا"
            and _tok(tokens, i-2)  == "به"
            and _tok(tokens, i-3)  == "شروع"
            and _tok(tokens, i+1)  == "درخت"
            and _tok(tokens, i+2)  == "می‌کند"
        ):
            _set_token(t,"آمدن")
            _set_lemma(t,"آمدن")
            _mark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) == "در"
            and _tok(tokens, i-1)  == "جلوی"
            and _tok(tokens, i+1)  == "است"
        ):
            _set_pos(t,"NOUN")
            continue

        if (
            _canon(_tok(tokens, i)) == "درخت"
            and _tok(tokens, i-1)  == "بالای"
            and _tok(tokens, i-2)  == "از"
            and _tok(tokens, i+1)  == "توپ"
            and _tok(tokens, i+2)  == "را"
        ):
            _unmark_ez(t)
            continue

        if (
            _canon(_tok(tokens, i)) in {"قهوه‌اش","قهوهاش"}
        ):
            _set_lemma(t,"قهوه")
            continue


        if (
            _canon(_tok(tokens, i)) == "اینجاهایی"
        ):
            _set_lemma(t,"اینجا")
            continue

        if (
            _canon(_tok(tokens, i)) in {"بیلچه‌اش","بیلچهاش"}
        ):
            _set_lemma(t,"بیلچه")
            continue

        if (
            _canon(_tok(tokens, i)) in {"آن","آن‌",'آ‌ن',"‌آن"}
        ):
            _set_lemma(t,"آن")
            _set_token(t,"آن")
            continue

        if (
            _canon(_tok(tokens, i)) in {"این","این‌",'‌این'}
        ):
            _set_lemma(t,"این")
            continue

        if (
            _canon(_tok(tokens, i)) in {"بچه","بچه‌",'‌بچه'}
        ):
            _set_lemma(t,"بچه")
            continue

        if (
            _canon(_tok(tokens, i)) == "عینک"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "آفتابی"
            and _tok(tokens, i+3)  == "است"
        ):
            _set_token(t,"عینکم")
            _set_pos(t,"NOUN")
            _unmark_ez(t)
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"عینک","role":"stem","morph_pos":"N"},
    {"form":"م","role":"PRON_CL","morph_pos":"PRON","person":"1","number":"Sing","case":"Poss"},
]
            continue

        if (
            _canon(_tok(tokens, i)) == "عینک"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "می‌آوردم"
        ):
            _set_token(t,"عینکم")
            _set_pos(t,"NOUN")
            _unmark_ez(t)
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"عینک","role":"stem","morph_pos":"N"},
    {"form":"م","role":"PRON_CL","morph_pos":"PRON","person":"1","number":"Sing","case":"Poss"},
]
            continue

        if (
            _canon(_tok(tokens, i)) == "عینک"
            and _tok(tokens, i+1)  == "هم"
            and _tok(tokens, i+2)  == "را"
            and _tok(tokens, i+3)  == "اگر"
            and _tok(tokens, i+4)  == "آورده"
            and _tok(tokens, i+5)  == "بودم"
        ):
            _set_token(t,"عینکم")
            _set_pos(t,"NOUN")
            _unmark_ez(t)
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"عینک","role":"stem","morph_pos":"N"},
    {"form":"م","role":"PRON_CL","morph_pos":"PRON","person":"1","number":"Sing","case":"Poss"},
]
            continue

        
        if (
            _canon(_tok(tokens, i)) == "یکیش"
        ):
            _set_token(t,"یکیش")
            _set_pos(t,"PRON")
            _set_lemma(t,"یکی")
            _unmark_ez(t)
            t["had_pron_clitic"] = True
            t["morph_segments"] = [
    {"form":"یکی","role":"stem","morph_pos":"PRON"},
    {"form":"ش","role":"PRON_CL","morph_pos":"PRON","person":"3","number":"Sing","case":"Poss"},
]
            continue

        if (
            _canon(_tok(tokens, i)) == "هم"
            and _tok(tokens, i-1)  == "عینکم"
            and _tok(tokens, i+1)  == "آفتابی"
            and _tok(tokens, i+2)  == "است"
        ):
            to_delete.add(i)
            continue

        if (
            _canon(_tok(tokens, i)) == "هم"
            and _tok(tokens, i-1)  == "عینکم"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "اگر"
            and _tok(tokens, i+3)  == "آورده"
            and _tok(tokens, i+4)  == "بودم"
        ):
            to_delete.add(i)
            continue

        if (
            _canon(_tok(tokens, i)) == "هم"
            and _tok(tokens, i-1)  == "عینکم"
            and _tok(tokens, i+1)  == "را"
            and _tok(tokens, i+2)  == "می‌آوردم"

        ):
            to_delete.add(i)
            continue

        if (
            _canon(_tok(tokens, i)) in {"دوتا","دو‌تا"}
        ):
            _set_lemma(t,"دو")
            continue

        if (
            _canon(_tok(tokens, i)) == "مانندی"
        ):
            _set_lemma(t,"مانند")
            continue

        if (
            _canon(_tok(tokens, i)) == "اولش"
        ):
            _set_lemma(t,"اول")
            continue

        if (
            _canon(_tok(tokens, i)) == "پیشش"
        ):
            _set_lemma(t,"پیش")
            continue

        if (
            _canon(_tok(tokens, i)) == "اولیش"
        ):
            _set_lemma(t,"اول")
            continue
        




        def _canon_tok_or_lem(t):
            return _canon(t.get("lemma") or t.get("tok") or "")

        # روی صندلی‌های عقب
        if (_canon(_tok(tokens, i)) == "روی"
            and _canon_tok_or_lem(tokens[i+1]) == "صندلی"
            and _canon_tok_or_lem(tokens[i+2]) == "عقب"):
            _mark_ez(tokens[i]); continue

        # روی پا وایمیسادیم  → match by lemma (ایستادن) or a tolerant surface
        if (_canon(_tok(tokens, i)) == "روی"
            and _canon_tok_or_lem(tokens[i+1]) == "پا"
            and (_canon_tok_or_lem(tokens[i+2]) == "ایستادن" or
                re.match(r"^می[\u200c\-]?ایست", _canon(_tok(tokens, i+2))) )):
            _mark_ez(tokens[i]); continue





        





    # حذف فیزیکی توکن‌های علامت‌خورده (هیچ چیز از آنها باقی نمی‌ماند)
    tokens = [t for j, t in enumerate(tokens) if j not in to_delete]

    # اگر قبلاً جایی tok را "" کرده‌ای، برای اطمینان اینها هم پاک شوند:
    tokens = [t for t in tokens if t.get("tok", "").strip() != ""]

    return tokens

def _apply_manual_person_number_overrides(tokens: list[dict]) -> list[dict]:
    """
    STRICT for verbs: only tweak AGR.Person/Number for exact surface forms in SURF_OVERRIDES.
    Also supports explicit clitic segmentation overrides (e.g., 'همه‌شان') via CLITIC_OVERRIDES.
    Idempotent: won't duplicate PRON_CL if already present.
    """
    SURF_OVERRIDES: dict[str, tuple[str, str]] = {
        _canon("بگویم"): ("1", "Sing"),
        _canon("بگوییم"): ("1", "Plur"),
        _canon("بیایم"): ("1", "Sing"),
        _canon("گویم"):  ("1", "Sing"),
        _canon("می‌گویم"):  ("1", "Sing"),
        _canon("بکند"):  ("1", "Sing"),
        _canon("می‌شوید"):  ("3", "Sing"),
        _canon("چید"):  ("3", "Sing"),
        _canon("درمی‌آید"):  ("3", "Sing"),
        _canon("می‌نشیند"):  ("3", "Sing"),
        _canon("بیاید"):  ("3", "Sing"),
        _canon("ببیند"):  ("3", "Sing"),
        _canon("بیایید"):  ("2", "Plur"),
        _canon("ببینید"):  ("2", "Plur"),
        _canon("می‌نشیند"):  ("3", "Sing"),
        _canon("خوابید"):  ("3", "Sing"),
        _canon("بشوید"):  ("3", "Sing"),
        _canon("می‌شکند"):  ("3", "Sing"),
        _canon("بکند"):  ("3", "Sing"),
        _canon("می‌گوییم"):  ("3", "Plur"),
        _canon("عینکم"):  ("1", "Sing"),
        _canon("بگوید"):  ("3", "Sing"),
        _canon("پرید"):  ("3", "Sing"),
        _canon("می‌چیند"):  ("3", "Sing"),
        _canon("بچیند"):  ("3", "Sing"),
        _canon("بزند"):  ("3", "Sing"),
        _canon("برود"):  ("3", "Sing"),
        _canon("بکند"):  ("3", "Sing"),


    }

    # استثناءهای پی‌بست: می‌توانید موارد مشابه را اضافه کنید
    CLITIC_OVERRIDES: dict[str, dict] = {
        _canon("همه‌شان"): {
            "lemma": "همه",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "PRON",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "همه",
            "stem_morph_pos": "PRON",
            "clitic_form": "شان",
            "person": "3",
            "number": "Plur",
            "case": "Poss",
        },

        _canon("عینکم"): {
            "lemma": "عینک",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "عینک",
            "stem_morph_pos": "NOUN",
            "clitic_form": "م",
            "person": "1",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("اولیش"): {
            "lemma": "اولی",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "اول",
            "stem_morph_pos": "NOUN",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        
        _canon("اولش"): {
            "lemma": "اول",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "اول",
            "stem_morph_pos": "NOUN",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("می‌فرستتش"): {
            "lemma": "فرستادن",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "VERB",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "فرستادن",
            "stem_morph_pos": "VERB",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("کیاش"): {
            "lemma": "کی",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "PRON",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "کی",
            "stem_morph_pos": "PRON",
            "clitic_form": "اش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("هرکدومش"): {
            "lemma": "هرکدام",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "PRON",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "هرکدام",
            "stem_morph_pos": "PRON",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("دوتاش"): {
            "lemma": "دوتا",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "دوتا",
            "stem_morph_pos": "N",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("اونجاش"): {
            "lemma": "آنجا",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "ADV",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "آنجا",
            "stem_morph_pos": "ADV",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("توپشون"): {
            "lemma": "توپ",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "توپ",
            "stem_morph_pos": "N",
            "clitic_form": "شون",
            "person": "3",
            "number": "Plur",
            "case": "Poss",
        },

        _canon("اینجاش"): {
            "lemma": "اینجا",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "ADV",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "اینجا",
            "stem_morph_pos": "ADV",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("یکی‌شان"): {
            "lemma": "یک",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "PRON",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "یکی",
            "stem_morph_pos": "PRON",
            "clitic_form": "شان",
            "person": "3",
            "number": "Plur",
            "case": "Poss",
        },

        _canon("باباش"): {
            "lemma": "بابا",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "بابا",
            "stem_morph_pos": "NOUN",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("مخیرم"): {
            "lemma": "مخیر",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "ADJ",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "مخیر",
            "stem_morph_pos": "ADJ",
            "clitic_form": "م",
            "person": "1",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("ابتداش"): {
            "lemma": "ابتدا",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "ابتدا",
            "stem_morph_pos": "NOUN",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },



        _canon("روبه‌روش"): {
            "lemma": "روبه‌رو",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "ADP",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "روبه‌رو",
            "stem_morph_pos": "ADP",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("آن‌طرفش"): {
            "lemma": "طرف",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "ADP",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "آن‌طرف",
            "stem_morph_pos": "ADP",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("همه‌اش"): {
            "lemma": "همه",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "PRON",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "همه",
            "stem_morph_pos": "PRON",
            "clitic_form": "اش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("برگ‌هایش"): {
            "lemma": "برگ",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "برگ",
            "stem_morph_pos": "N",
            "clitic_form": "یش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("اینش"): {
            "lemma": "این",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "این",
            "stem_morph_pos": "N",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("نصفش"): {
            "lemma": "نصف",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "NOUN",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "نصف",
            "stem_morph_pos": "N",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("کسیش"): {
            "lemma": "کس",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "PRON",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "کس",
            "stem_morph_pos": "PRON",
            "clitic_form": "یش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },

        _canon("دیگه‌ش"): {
            "lemma": "دیگر",          # اگر نمی‌خواهید lemma تغییر کند، این خط را حذف کنید
            "pos":   "ADJ",         # اگر نمی‌خواهید POS تغییر کند، این خط را حذف کنید
            "stem_form": "دیگر",
            "stem_morph_pos": "ADJ",
            "clitic_form": "ش",
            "person": "3",
            "number": "Sing",
            "case": "Poss",
        },
        
        
        # مثال‌های دیگر: آن‌طرفش
        # _canon("کنارش"): {"lemma":"کنار","pos":"ADP","stem_form":"کنار","stem_morph_pos":"ADP","clitic_form":"ش","person":"3","number":"Sing","case":"Poss"}
    }
    n = len(tokens)
    for i, t in enumerate(tokens):
        key = _canon(t.get("tok", "") or "")
        if key == _canon("باید"):
            t["pos"] = "AUX"        # اطمینان از AUX بودن
            t["lemma"] = "باید"     # بدون ریشه‌سازی؛ خودِ «باید»
            t["morph_segments"] = []  # هرچه هست پاک کن (stem/AGR/...)
            continue

        nxt_key  = _canon(tokens[i+1].get("tok", "") or "") if i + 1 < n else ""
        
        if i > 0:
            prev_key = _canon(tokens[i-1].get("tok", "") or "")


            if key == _canon("فروشی") and prev_key == _canon("مس"):
                t["pos"] = "ADJ"
                t["lemma"] = "فروشی"
                t["morph_segments"] = []

            if key == _canon("تاشون") and prev_key in {"دو","سه"}:
                t["pos"] = "PART"
                t["lemma"] = "تا"
                t["morph_segments"] = [
    {"form":"دوتا","role":"stem","morph_pos":"PART"},
    {"form":"شون","role":"PRON_CL","morph_pos":"PART","person":"3","number":"Plur","case":"Poss"},
]
            
            if key == _canon("کنار") and prev_key == _canon("در") and nxt_key == _canon("آن"):
                t["pos"] = "NOUN"
                t["lemma"] = "کنار"

            if key == _canon("تو") and prev_key == _canon("رفته"):
                t["pos"] = "ADP"
                t["lemma"] = "تو"
                t["morph_segments"] = []

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1)  == "به"
                and _tok(tokens, i+2)  == "اصطلاح"
                and _tok(tokens, i+3)  == "آن"
                and _tok(tokens, i+4)  == "ظرفشویی"
            ):
                t["pos"] = "ADP"
                continue

            if (
                _canon(_tok(tokens, i)) == "پایین"
                and _tok(tokens, i-1)  == "می‌پرد"
                and _tok(tokens, i-2)  == "گربه"
            ):
                t["pos"] = "ADV"
                continue

            if (
                _canon(_tok(tokens, i)) == "آن"
                and _tok(tokens, i+1)  == "کتاب"
                and _tok(tokens, i+2)  == "فردوسی"
            ):
                t["pos"] = "DET"
                continue

            if key == _canon("تو") and prev_key == _canon("می‌آید"):
                t["pos"] = "ADP"
                t["lemma"] = "تو"
                t["morph_segments"] = []

            if (
                _canon(_tok(tokens, i)) in {"چرخ‌دستی","چرخدستی"}
            ):
                _set_pos(t,"NOUN")
                t["morph_segments"] = []
                continue

            if (
                _canon(_tok(tokens, i)) in {"چرخ‌دستی","چرخدستی"}
                and _tok(tokens, i-1)  == "خب"
            ):
                _set_pos(t,"NOUN")
                t["morph_segments"] = []
                continue

            if key == _canon("بعد") and prev_key == _canon("تصویر"):
                t["pos"] = "ADJ"
                t["lemma"] = "بعد"
                t["morph_segments"] = []
            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "دوش"
                and _tok(tokens, i+2) == "است"
            ):
                _mark_ez(t)
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "همان"
                and _tok(tokens, i+2) == "صندلی"
            ):
                _mark_ez(t)
                continue

            if (
                _canon(_tok(tokens, i)) == "بالای"
                and _tok(tokens, i+1) == "کتابخانه‌اش"
                and _tok(tokens, i+2) == "هست"
            ):
                t["pos"] = "ADP"
                _mark_ez(t)
                continue

            
            if (
                _canon(_tok(tokens, i)) == "بالای"
                and _tok(tokens, i+1) == "چه"
                and _tok(tokens, i+2) == "هست"
            ):
                t["pos"] = "ADP"
                _mark_ez(t)
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) in {"دیوار","سفره","درخت","الاغ","میز"}
            ):
                _mark_ez(t)
                continue

            if (
                _canon(_tok(tokens, i)) == "روی"
                and _tok(tokens, i+1) == "زیر"
                and _tok(tokens, i+2) == "چانه‌اش"

            ):
                _mark_ez(t)
                continue

            
            if (
                _canon(_tok(tokens, i)) in {"نقش‌ونگارهای","نقشونگارهای"}
            ):
                _set_lemma(t,"نقش‌ونگار")
                continue

            if (
                _canon(_tok(tokens, i)) == "وری"
                and _tok(tokens, i-1) == "یک"
                and _tok(tokens, i+1) == "کجش"

            ):
                _set_pos(t,"ADV")
                _unmark_ez(t)
                continue

            if key == _canon("کردی") and prev_key == _canon("لباس"):
                t["pos"] = "ADJ"
                t["lemma"] = "کردی"
                t["morph_segments"] = []

            if key == _canon("این") and prev_key == _canon("از") and nxt_key == _canon("تصویر"):
                t["pos"] = "DET"
                t["lemma"] = "این"
                t["morph_segments"] = []

            if key == _canon("روی") and prev_key == _canon("از") and nxt_key == _canon("آن"):
                t["pos"] = "ADP"

            if key == _canon("هرکدوم"):
                t["pos"] = "PRON"
                t["lemma"] = "هرکدام"
                t["morph_segments"] = []
                

            if key == _canon("یعنی"):
                t["morph_segments"] = []

            if key == _canon("یکی"):
                t["morph_segments"] = []

            if key == _canon("درواقع"):
                t["morph_segments"] = []

            if key == _canon("انگاری"):
                t["morph_segments"] = []

            if key == _canon("دنبال"):
                t["morph_segments"] = []

            if key == _canon("بعدش"):
                t["morph_segments"] = []

            if key == _canon("چه"):
                t["morph_segments"] = []

            if key == _canon("هرکدومش"):
                t["morph_segments"] = []

        if key == _canon("واژگون"):
            t["pos"] = "ADJ"        # اطمینان از AUX بودن
            t["lemma"] = "واژگون"     # بدون ریشه‌سازی؛ خودِ «باید»
            t["feats"] = []
            t[""] = []  # هرچه هست پاک کن (stem/AGR/...)
            continue
        if key == _canon("برگی"):
            t["pos"] = "NOUN"        # اطمینان از AUX بودن
            t["lemma"] = "برگ"     # بدون ریشه‌سازی؛ خودِ «باید»
            t["feats"] = []
            t["morph_segments"] = []  # هرچه هست پاک کن (stem/AGR/...)
            continue
        if key == _canon("موز"):
            t["pos"] = "NOUN"        # اطمینان از AUX بودن
            t["lemma"] = "موز"     # بدون ریشه‌سازی؛ خودِ «باید»
            t["feats"] = []
            t[""] = []  # هرچه هست پاک کن (stem/AGR/...)
            continue
        if key == _canon("جهاند"):
            t["tok"] = "جهان"
            t["pos"] = "NOUN"        # اطمینان از AUX بودن
            t["lemma"] = "جهان"     # بدون ریشه‌سازی؛ خودِ «باید»
            t["feats"] = []
            t[""] = []  # هرچه هست پاک کن (stem/AGR/...)
            continue

        if key == _canon("چراغهاش"):
            t["tok"] = "چراغ‌هایش"
            t["pos"] = "NOUN"        # اطمینان از AUX بودن
            t["lemma"] = "چراغ"     # بدون ریشه‌سازی؛ خودِ «باید»
            t["feats"] = []
            t[""] = []  # هرچه هست پاک کن (stem/AGR/...)
            continue

        if key == _canon("خانهی"):
            t["tok"] = "خانه‌ی"
            t["pos"] = "NOUN"        # اطمینان از AUX بودن
            t["lemma"] = "خانه"     # بدون ریشه‌سازی؛ خودِ «باید»
            t["feats"] = []
            t[""] = []  # هرچه هست پاک کن (stem/AGR/...)
            continue

        if key == _canon("هی"):
            t["tok"] = "هی"
            t["pos"] = "ADV"        # اطمینان از AUX بودن
            t["lemma"] = "هی"     # بدون ریشه‌سازی؛ خودِ «باید»
            t["feats"] = []
            t[""] = []  # هرچه هست پاک کن (stem/AGR/...)
            continue

        # 1) اعمال استثناءهای پی‌بست (روی PRON/ADP/… هم کار می‌کند)
        if key in CLITIC_OVERRIDES:
            spec = CLITIC_OVERRIDES[key]

            # اگر قبلاً PRON_CL وجود دارد، چیزی اضافه نکن
            segs = t.get("morph_segments") or []
            has_pron_cl = any(seg.get("role") == "PRON_CL" for seg in segs)


            if not has_pron_cl:
                t["morph_segments"] = [
                    {
                        "form": spec["stem_form"],
                        "role": "stem",
                        "morph_pos": spec.get("stem_morph_pos") or t.get("pos", "")
                    },
                    {
                        "form": spec["clitic_form"],
                        "role": "PRON_CL",
                        "Person": spec["person"],
                        "Number": spec["number"],
                        "Case": spec["case"],
                    },
                ]

            # پرچم‌ها
            t["had_clitic"] = True
            t["had_pron_clitic"] = True
            if spec.get("number", "").lower().startswith("plur"):
                t["had_plural_suffix"] = True

            # در صورت تمایل lemma/POS را اصلاح کن
            if "lemma" in spec:
                t["lemma"] = spec["lemma"]
            if "pos" in spec:
                t["pos"] = spec["pos"]

            # از این مورد بگذر و به توکن بعدی برو
            continue

        # 2) همان منطق قبلی برای افعال (فقط AGR را دست می‌زنیم)
        if not str(t.get("pos", "")).startswith("VERB"):
            continue
        if key not in SURF_OVERRIDES:
            continue

        per, num = SURF_OVERRIDES[key]
        segs = t.get("morph_segments") or []
        changed = False
        for seg in segs:
            if seg.get("role") == "AGR":
                seg["Person"] = per
                seg["Number"] = num
                changed = True
        if changed:
            t["morph_segments"] = segs

    return tokens





# =============================================================================
# 6) Main normaliser class
# =============================================================================

class Normaliser:
    def __call__(self, text: str) -> dict:
        raw = text.rstrip("\n")

        # ----- STAGE 1: pre-tokenization canonicalization -----
        x = canon_chars(raw)
        x = _normalize_inshallah(x)          # <— add here
        x = _apply_phrase_replacements(x)
        x = _apply_token_replacements(x)
        # Attach any stand-alone pronominal clitic to its host
        _PRON = r'(?:ام|ات|اش|ایم|اید|اند|مان|تان|شان|مون|تون|شون|م|ت|ش)'
        x = re.sub(rf'([^\s]+)\s+({_PRON})\b', r'\1\2', x)

        x = _pretoken_colloquial(x)

        # Avoid «…هم» fused on clitic-hosts → split: «…» + «هم»
        fused_hem_re = re.compile(r'([^\s\u200c]+(?:[\u200c])?(?:م|ت|ش|مان|تان|شان|مون|تون|شون))(هم)\b')
        x = fused_hem_re.sub(r'\1 \2', x)

        x = _post_rules(x)
        x = _fix_raftoamad_spacing(x)
        # underscores used in some raw files (e.g., past comp) → visible space
        x = re.sub(r'(?<=\S)_(?=\S)', ' ', x)

        # ----- STAGE 2: core tokenization -----
        tokens = dadma_tokenise(
            x, merge_clitics=True, merge_mi=True, non_clitic_words=NON_CLITIC_WORDS
        )
        # ----- STAGE 3: post-tokenization refinement (order matters) -----
        tokens = _undo_false_zwnj_on_lexemes (tokens)
        tokens = _merge_bare_pron_tokens(tokens)
        tokens = _apply_manual_pos_overrides(tokens)
        tokens = _retag_closed(tokens)
        tokens = _retag_contextual(tokens)
        tokens = _retag_no_particle(tokens)
        tokens = _enforce_progressive_aux_postud(tokens)      # <-- NEW early pass (idempotent)
        tokens = _fix_zwnj_before_pron_on_tokens(tokens)
        tokens = _final_idempotent_repairs(tokens)
        tokens = _final_batch2_repairs(tokens)
        tokens = _final_batch2_hard_patch(tokens)
        tokens = _fix_heh_ezafe_graph(tokens)
        tokens = _fuse_simple_compounds(tokens)
        tokens = _normalize_bare_heh_copula(tokens)  # NEW
        tokens = _create_morph_segments(tokens)
        tokens = _retag_yfinal_nominal(tokens)       # NEW
        tokens = _normalize_hal_finite_to_masdar(tokens)  # NEW
        tokens = _retag_yfinal_nominal(tokens)
        tokens = _normalize_hal_finite_to_masdar(tokens)
        tokens = _prevent_phantom_clitic(tokens)
        tokens = _batch_scrub_tail_m(tokens)
        tokens = _apply_manual_person_number_overrides(tokens)
        # Derive flags from morphology (do not let ZWNJ drive counts)
        tokens = _ensure_clitic_flags(tokens)
        tokens = _sync_had_flags_to_morph(tokens)
        
  
        

        # Lemma overrides + safety fallback
        tokens = [_override_lemma(t) for t in tokens]
        tokens = [_noun_lemma_fallback(t) for t in tokens]

        # UD mapping + structural postprocess
        tokens =  map_sentence_to_ud(tokens)
        tokens = _enforce_progressive_aux_postud(tokens)   # <— add here
        tokens = _normalize_bala_adp(tokens)
        tokens = _normalize_paeen_pehlu_adp(tokens)   # <- NEW
        tokens = _normalize_jelo_adp(tokens)
        tokens = _normalize_lab_adp(tokens)      # NEW
        tokens = _normalize_ruberoo_adp(tokens) 
        tokens = _ensure_ezafe_feature_consistency(tokens) 
        tokens = _enforce_ezafe_on_postpositions(tokens)
        tokens = _strip_ezafe_from_adps(tokens)
        tokens = _sync_had_ez_to_features(tokens)
        tokens = _ezafe_postprocess_ud(tokens)
        tokens = _batch_fix_dar_hali(tokens)
        tokens = _batch_ezafe_pair_scrub(tokens)
        tokens = _batch_relational_bala_final_pass(tokens)
        tokens = _fix_lvc_ezafe(tokens)
        tokens = _fix_misplaced_ezafe(tokens)
        tokens = _batch_force_masdar_after_hal(tokens)   # <— add here
        tokens = _force_morphpos_for_verbs(tokens)

        tokens = _force_morphpos_from_pos(tokens)

        tokens = _batch_force_ezafe_on_bala(tokens)
        tokens = _strip_balay_before_punct(tokens)         # <— add here

        tokens = _scrub_sentence_final_bala(tokens)   # fixes: تابلوِ بالایِ تلویزیون
        tokens = _batch_fix_mizgard(tokens) 
        tokens = _batch_fix_haman(tokens)
        tokens = _batch_fix_yad_typo(tokens) 
        tokens = _strip_ezafe_from_adps(tokens) 
        tokens = _apply_manual_pos_overrides(tokens)      # your simple if/continue rules
        tokens = _force_morphpos_from_pos(tokens)         # keep morph_segments consistent
        tokens = _apply_manual_lemma_overrides (tokens)
        tokens = _apply_manual_ezafe_overrides(tokens)
        tokens = _apply_token_lemma_overrides(tokens)  # your simple if/continue rules
        tokens = _sync_had_ez_to_features(tokens) 
        tokens = _force_morphpos_from_pos(tokens)



        # ----- STAGE 4: final output string -----
        final_norm = " ".join(t['tok'] for t in tokens)
        return {"raw": raw, "norm": final_norm, "tokens": tokens}

# =============================================================================
# 7) CLI
# =============================================================================

def normalise_file(path: Path, out_dir: Path, norm: Normaliser):
    out_dir.mkdir(parents=True, exist_ok=True)
    txt  = path.read_text(encoding="utf-8")
    data = norm(txt)
    out_f = out_dir / (path.stem + ".json")
    out_f.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")

def normalise_dir(in_dir: Path, out_dir: Path):
    norm = Normaliser()
    files = [normalise_file(p, out_dir, norm) for p in sorted(in_dir.glob("*.txt"))]
    print(f"[✓] Normalised {len(files)} files → {out_dir}")

def _cli():
    ap = argparse.ArgumentParser(description="Farsi normalization + UD postprocess")
    ap.add_argument("--in",  dest="indir",  required=True, help="Input folder with raw .txt files")
    ap.add_argument("--out", dest="outdir", required=True, help="Output folder for processed .json files")
    args = ap.parse_args()
    normalise_dir(Path(args.indir), Path(args.outdir))

if __name__ == "__main__":
    _cli()
