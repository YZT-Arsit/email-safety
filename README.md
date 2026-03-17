# EmailSafety Closed Loop

A reproducible email risk classification project built from 24k unlabeled mail logs, designed for large-model algorithm and LLM application interviews.

## What This Project Solves
This repository starts from a realistic constraint: a mail security dataset with no labels, heterogeneous fields, and noisy content. The project turns that into a complete learning loop:

- define a 5-class email risk taxonomy
- build Gold v1 and Gold v2 from seed annotation and targeted relabeling
- use the full 24k unlabeled corpus for domain-adaptive pretraining (DAPT / MLM)
- generate high-precision trusted silver through multi-teacher consensus
- compare Gold-only and semi-supervised downstream training

Target task: 5-way email risk classification.

Taxonomy:
- `advertisement`
- `phishing`
- `impersonation`
- `malicious_link_or_attachment`
- `black_industry_or_policy_violation`

## Project Highlights
- **Realistic weak-to-strong data loop**: unlabeled logs -> taxonomy -> Gold -> trusted silver -> semi-supervised training.
- **Offline-first model workflow**: all Transformer checkpoints are loaded from local ModelScope paths, not Hugging Face online downloads.
- **Two ways to use unlabeled data**:
  - DAPT with MLM on 24k raw emails
  - consensus trusted silver for train-only augmentation
- **Interview-friendly rigor**: Gold-only valid/test split is kept clean while silver stays in train with lower weight.

## Closed Loop
```text
24k unlabeled mail logs
  -> cleaned parsing + taxonomy definition
  -> Gold v1 (1,000)
  -> targeted relabeling
  -> Gold v2 (1,491)
  -> multilingual BERT downstream baseline
  -> DAPT on full unlabeled corpus
  -> DAPT downstream comparison
  -> multi-teacher consensus trusted silver (127)
  -> Gold + trusted silver semi-supervised training
  -> final result summary and interview materials
```

## Core Results
Current key outputs:
- Gold v2: `1491` samples
- MLM corpus retained rows: `23717 / 24000`
- Trusted silver: `127`
- Best baseline: `text_only_lr`, macro F1 = `0.6325`
- Best multilingual BERT: `multilingual_bert_dapt`, test macro F1 = `0.6088`
- Best semi-supervised setting: `gold_v2_only__dapt_multilingual_bert`, test macro F1 = `0.6122`

Final artifacts kept in GitHub version:
- [outputs/final_summary/final_closed_loop_results.csv](/Users/Hoshino/Documents/emailsafety/outputs/final_summary/final_closed_loop_results.csv)
- [outputs/final_summary/final_closed_loop_summary.md](/Users/Hoshino/Documents/emailsafety/outputs/final_summary/final_closed_loop_summary.md)
- [outputs/final_summary/final_interview_bullets.md](/Users/Hoshino/Documents/emailsafety/outputs/final_summary/final_interview_bullets.md)

## Repository Structure
High-level layout:
- `configs/`: reproducible config files
- `data/`: annotation assets, processed splits, MLM corpus
- `docs/`: taxonomy, project summary, interview notes
- `models/`: local model checkpoints, not committed to GitHub
- `outputs/final_summary/`: final public-facing results kept in repo
- `scripts/`: final closed-loop entrypoints
- `src/`: reusable Python package code
- `archive/`: legacy scripts and configs kept for reference, including earlier baseline/fusion stages

More detail: [PROJECT_STRUCTURE.md](/Users/Hoshino/Documents/emailsafety/PROJECT_STRUCTURE.md)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Minimal Reproduction Path
```bash
python scripts/download_model_from_modelscope.py \
  --model-id AI-ModelScope/bert-base-multilingual-cased \
  --cache-dir models

python scripts/build_mlm_corpus.py \
  --input-path spam_email_data.log \
  --output-path data/mlm_corpus/mail_corpus.txt \
  --stats-json data/mlm_corpus/mail_corpus_stats.json

python scripts/train_dapt_mlm.py \
  --model-dir models/bert-base-multilingual-cased \
  --corpus-txt data/mlm_corpus/mail_corpus.txt \
  --output-dir outputs/dapt_multilingual_bert

python scripts/run_multilingual_dapt_comparison.py \
  --gold-csv data/annotation/gold/gold_v2.csv \
  --base-model-dir models/bert-base-multilingual-cased \
  --dapt-model-dir outputs/dapt_multilingual_bert/final_model

python scripts/summarize_final_closed_loop.py \
  --output-dir outputs/final_summary
```

For the full end-to-end run order, commands, inputs, outputs, and low-memory settings, see [RUNBOOK.md](/Users/Hoshino/Documents/emailsafety/RUNBOOK.md).

## Result Files
- Final public summary: `outputs/final_summary/`
- Local-only experiment outputs: `outputs/formal_baselines/`, `outputs/multilingual_dapt_comparison/`, `outputs/semi_supervised_comparison/`
- Local-only model cache/checkpoints: `models/`, `outputs/dapt_multilingual_bert/`

## Limitations
- Trusted silver is intentionally conservative and currently biased toward the dominant class.
- Long-tail classes such as `impersonation` and `malicious_link_or_attachment` remain harder than the head class.
- The project prioritizes reproducibility and closed-loop data usage over aggressive hyperparameter search.

## Future Work
- stronger multilingual backbone or encoder-only long-text variant
- better per-class silver acceptance policies
- richer text-structure fusion under the same clean evaluation protocol
