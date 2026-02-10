import unicodedata, re

_AR2FA = {
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ؤ": "و",
    "\u06CC": "ی",  # ARABIC LETTER FARSI YEH
}

_DUP_ZWNJ = re.compile(r"\u200c{2,}")        # collapse duplicate ZWNJ
_DIACRITIC = re.compile(r"[\u0610-\u061A\u064B-\u065F]")

def canon_chars(text: str, strip_diacritics: bool = True) -> str:
    """Unicode NFC + Arabic‑to‑Persian letters + zwnj clean."""
    text = unicodedata.normalize("NFC", text)
    for src, tgt in _AR2FA.items():
        text = text.replace(src, tgt)
    text = _DUP_ZWNJ.sub("\u200c", text)
    if strip_diacritics:
        text = _DIACRITIC.sub("", text)
    return text
