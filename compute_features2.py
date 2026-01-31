import os
import json
import math
from collections import Counter, defaultdict
import statistics as stats
import csv

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

INPUT_DIR = r"C:\Users\Fatiima\Desktop\voices\thesis_2\persian_norm\norm775"
OUTPUT_CSV = r"C:\Users\Fatiima\Desktop\voices\thesis_2\features_all_metrics_updated2.csv"

# Define POS tags
CONTENT_POS = {"NOUN", "VERB","NUM", "ADJ", "ADV", "PROPN"}
FUNCTION_POS = {
    "PRON", "DET", "ADP", "AUX", "PART", "CCONJ", "SCONJ",
    "INTJ", "PUNCT", "SYM"
}

# Common Persian Light Verbs / High Frequency Verbs (Lemmas)
# Tracking these helps detect "Empty Speech" where patients rely on generic verbs.
LIGHT_VERB_LEMMAS = {
    "کردن",   # kardan (to do/make)
    "شدن",    # shodan (to become)
    "بودن",   # budan (to be)
    "است",    # ast (is)
    "داشتن",  # dashtan (to have)
    "دادن",   # dadan (to give)
    "زدن",    # zadan (to hit/strike - used in many complex predicates)
    "گرفتن"   # gereftan (to take - often used as light verb)
}

FILLER_LEMMAS = {"م", "ام", "آه", "اِ", "اِه", "یعنی", "مثلا"}
AND_LEMMA = "و"

# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------

def safe_log(x, base=math.e):
    if x <= 0: return 0.0
    return math.log(x, base)

def shannon_entropy(probabilities, base=2.0):
    h = 0.0
    for p in probabilities:
        if p > 0: h -= p * math.log(p, base)
    return h

# ---------------------------------------------------------
# LEXICAL MEASURES
# ---------------------------------------------------------

def lexical_basic_counts(tokens):
    lemmas = []
    for t in tokens:
        if t.get("pos") == "PUNCT":
            continue
        lemma = t.get("lemma", t.get("tok"))
        lemma = str(lemma).strip()
        if not lemma: continue
        lemmas.append(lemma)

    N = len(lemmas)
    freq = Counter(lemmas)
    V = len(freq)
    return N, V, freq, lemmas

def compute_lexical_diversity(freq, N):
    V = len(freq)
    if N == 0 or V == 0:
        return {"TTR": 0.0, "Honore_R": 0.0, "Brunet_W": 0.0, "Lexical_Entropy": 0.0}

    f_of_f = Counter(freq.values())
    V1 = f_of_f.get(1, 0)
    V2 = f_of_f.get(2, 0)

    ttr = V / N
    honore = 100.0 * safe_log(N) / (1.0 - (V1 / V)) if V1 < V else 0.0
    brunet_W = N ** (V ** (-0.172))
    
    probs = [c / N for c in freq.values()]
    lex_entropy = shannon_entropy(probs, base=2.0)

    return {
        "TTR": ttr,
        "Honore_R": honore,
        "Brunet_W": brunet_W,
        "Lexical_Entropy": lex_entropy
    }

def compute_MTLD(lemmas, ttr_threshold=0.72):
    if not lemmas: return 0.0
    def _mtld_factor_count(seq):
        factors = 0
        token_count = 0
        types = set()
        for tok in seq:
            token_count += 1
            types.add(tok)
            ttr = len(types) / token_count
            if ttr <= ttr_threshold:
                factors += 1
                token_count = 0
                types = set()
        if token_count > 0:
            factors += (1 - (ttr_threshold - len(types) / token_count)) / (1 - ttr_threshold)
        return factors
    
    fwd = _mtld_factor_count(lemmas)
    rev = _mtld_factor_count(list(reversed(lemmas)))
    avg = (fwd + rev) / 2.0 if (fwd + rev) > 0 else 1.0
    return len(lemmas) / avg

# ---------------------------------------------------------
# POS, NOUN/VERB RATIOS & MORPHOLOGY
# ---------------------------------------------------------

def compute_pos_and_morphology(tokens):
    pos_counts = Counter()
    
    # Storage for Morphology Analysis
    # Structure: {'lemma': {'form1', 'form2'}}
    verb_morphology = defaultdict(set) 
    noun_morphology = defaultdict(set)
    
    light_verb_tokens = 0
    total_non_punct = 0
    
    content_words = 0
    function_words = 0
    
    for t in tokens:
        pos = t.get("pos")
        tok = str(t.get("tok", "")).strip()
        lemma = str(t.get("lemma", tok)).strip()
        
        if pos == "PUNCT": continue
        
        total_non_punct += 1
        pos_counts[pos] += 1
        
        if pos in CONTENT_POS: content_words += 1
        if pos in FUNCTION_POS: function_words += 1
        
        # 1. Track Morphology (Forms per Lemma)
        if pos == "VERB" or pos == "AUX":
            verb_morphology[lemma].add(tok)
            # Check for Light Verb Usage
            if lemma in LIGHT_VERB_LEMMAS:
                light_verb_tokens += 1
                
        if pos == "NOUN" or pos == "PROPN":
            noun_morphology[lemma].add(tok)

    # --- Noun to Verb Ratio ---
    # We include PROPN in nouns and AUX in verbs for a broad grammatical picture,
    # or you can strictly use 'NOUN' and 'VERB'. Here we use strict NOUN/VERB for standard N/V ratio.
    noun_strict = pos_counts.get("NOUN", 0)
    verb_strict = pos_counts.get("VERB", 0)
    
    if verb_strict > 0:
        nv_ratio = noun_strict / verb_strict
    else:
        nv_ratio = 0.0 # Or noun_strict if you prefer, but 0 is safer for stats
        
    # --- Morphological Richness (Inflectional Diversity) ---
    # How many different surface forms (tok) exist on average for each lemma?
    
    # For Verbs
    num_verb_lemmas = len(verb_morphology)
    total_verb_forms = sum(len(forms) for forms in verb_morphology.values())
    
    if num_verb_lemmas > 0:
        avg_forms_per_verb = total_verb_forms / num_verb_lemmas
    else:
        avg_forms_per_verb = 0.0
        
    # For Nouns (usually lower in Persian unless plural/definite markers are attached to token)
    num_noun_lemmas = len(noun_morphology)
    total_noun_forms = sum(len(forms) for forms in noun_morphology.values())
    
    if num_noun_lemmas > 0:
        avg_forms_per_noun = total_noun_forms / num_noun_lemmas
    else:
        avg_forms_per_noun = 0.0

    # --- Light Verb Rate ---
    light_verb_rate = (light_verb_tokens / total_non_punct) if total_non_punct > 0 else 0.0

    return {
        "POS_NOUN": noun_strict,
        "POS_VERB": verb_strict,
        "Noun_Verb_Ratio": nv_ratio,
        "Avg_Forms_Per_Verb_Lemma": avg_forms_per_verb,
        "Avg_Forms_Per_Noun_Lemma": avg_forms_per_noun,
        "Light_Verb_Rate": light_verb_rate,
        "Content_Function_Ratio": content_words / function_words if function_words > 0 else 0
    }

# ---------------------------------------------------------
# IDEA DENSITY
# ---------------------------------------------------------

def compute_idea_density(tokens):
    content = 0
    total = 0
    
    # Ensure you have access to your light verb list here
    # (Assuming LIGHT_VERB_LEMMAS is defined globally as in your config)
    
    for t in tokens:
        # 1. Skip punctuation/symbols for both counts
        if t.get("pos") in ["PUNCT", "SYM", "SPACE"]: 
            continue
            
        # 2. Increment total word count (denominator)
        total += 1
        
        # 3. Check for Content Words
        if t.get("pos") in CONTENT_POS:
            lemma = str(t.get("lemma", t.get("tok", ""))).strip()
            
            # --- CRITICAL CHANGE ---
            # If it is a verb, but it is a LIGHT verb, treat it as a function word (ignore it).
            if t.get("pos") == "VERB" and lemma in LIGHT_VERB_LEMMAS:
                continue
            # -----------------------

            content += 1

    ratio = content / total if total > 0 else 0
    
    return {
        "Idea_Density_per_10words": ratio * 10.0,
        "Content_Word_Ratio": ratio
    }

# ---------------------------------------------------------
# GRAPH FEATURES
# ---------------------------------------------------------

def compute_graph_features(tokens):
    lemmas = [str(t.get("lemma", t.get("tok", ""))).strip() for t in tokens if t.get("pos") != "PUNCT"]
    lemmas = [l for l in lemmas if l]
    
    if not lemmas: return {"Graph_Nodes": 0, "Graph_Edges": 0, "Graph_Avg_OutDegree": 0.0}

    nodes = set(lemmas)
    edges = set()
    out_degree = defaultdict(int)

    for i in range(len(lemmas) - 1):
        u, v = lemmas[i], lemmas[i+1]
        if (u, v) not in edges:
            edges.add((u, v))
            out_degree[u] += 1

    num_nodes = len(nodes)
    avg_out = sum(out_degree.values()) / num_nodes if num_nodes > 0 else 0

    return {
        "Graph_Nodes": num_nodes,
        "Graph_Edges": len(edges),
        "Graph_Avg_OutDegree": avg_out
    }

# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------

def compute_features_for_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = os.path.basename(json_path)
    participant_id = os.path.splitext(filename)[0]
    tokens = data.get("tokens", [])
    norm_text = data.get("norm", "")

    # 1. Lexical
    N, V, freq, lemmas = lexical_basic_counts(tokens)
    lex_feats = compute_lexical_diversity(freq, N)
    mtld = compute_MTLD(lemmas)

    # 2. POS, N/V Ratio, Morphology
    pos_morph_feats = compute_pos_and_morphology(tokens)

    # 3. Idea Density
    idea_feats = compute_idea_density(tokens)

    # 4. Graph
    graph_feats = compute_graph_features(tokens)
    
    # 5. Sentence Proxies (Simplified)
    # Using simple sentence count based on punctuation
    sentences = [t for t in tokens if t.get("pos") == "PUNCT" and t.get("tok") in {".", "؟", "!", ";"}]
    num_sentences = len(sentences) if len(sentences) > 0 else 1
    mean_sent_len = N / num_sentences

    return {
        "Participant": participant_id,
        "Num_Tokens": N,
        "Num_Unique_Lemmas": V,
        "Mean_Sent_Length": mean_sent_len,
        **lex_feats,
        "MTLD": mtld,
        **pos_morph_feats,
        **idea_feats,
        **graph_feats
    }

def main():
    json_files = [os.path.join(INPUT_DIR, fn) for fn in os.listdir(INPUT_DIR) if fn.lower().endswith(".json")]
    
    if not json_files:
        print("No JSON files found.")
        return

    all_data = []
    print(f"Processing {len(json_files)} files...")
    
    for path in sorted(json_files):
        try:
            feats = compute_features_for_json(path)
            all_data.append(feats)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if all_data:
        keys = ["Participant"] + sorted([k for k in all_data[0].keys() if k != "Participant"])
        
        with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_data)
            
        print(f"Done! Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()