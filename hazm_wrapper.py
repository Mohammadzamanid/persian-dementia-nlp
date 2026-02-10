from hazm import Normalizer

hazm_norm = Normalizer(
    persian_numbers=True,
    remove_diacritics=False,
    correct_spacing=False,   # turn OFF Hazm spacing fixes
    seperate_mi=False        # keep «می/نمی» intact
)

def hazm_normalise(text: str) -> str:
    return hazm_norm.normalize(text)