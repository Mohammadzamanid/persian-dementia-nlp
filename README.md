# Persian Dementia NLP (Connected Speech) — LII / Robust Inference / Internal Validation

This repository contains the analysis code and *derived, de-identified* feature tables for a Persian (Farsi) connected-speech study across:
- Healthy controls
- Mild Cognitive Impairment (MCI)
- Mild Alzheimer disease (mild AD)

Primary aim: evaluate **Language Informativeness Index (LII)** and related speech/language measures under robust covariate-adjusted inference.
Secondary aim: exploratory internal-validation prediction (multiclass), reported transparently.

> **Important (privacy):** Raw transcripts and audio are not included in this public repository because they are sensitive human-participant data. Derived feature tables and locked statistical outputs are provided to enable reproducible results without re-identification risk.

## What’s in this repository

Recommended structure (example):
- `data/derived/`  
  De-identified derived feature tables used in the paper (e.g., analysis-ready participant-level table).
- `results/locked/`  
  Locked model outputs (HC3 robust OLS omnibus tests, contrasts, multiplicity-corrected results).
- `scripts/`  
  Python scripts used for analysis and figure/table generation.
- `references/lii/`  
  Picture reference descriptions used to compute LII (non-participant text; safe to share).
- `docs/`  
  Data dictionary and reproducibility notes.

## Reproducibility (quickstart)

### 1) Create a clean environment
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
python scripts/<your_main_analysis_script>.py

If you use this code, please cite the repository. See CITATION.cff
