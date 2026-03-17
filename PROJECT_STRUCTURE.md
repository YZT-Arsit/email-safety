# PROJECT_STRUCTURE

## Top-Level Layout
- `README.md`: GitHub homepage and project overview
- `RUNBOOK.md`: full reproduction manual
- `PROJECT_STRUCTURE.md`: repository layout and publishing guidance
- `requirements.txt`: Python dependencies
- `.gitignore`: files and directories excluded from GitHub
- `configs/`: active config files used by the final closed loop
- `data/`: annotation assets, processed splits, corpus artifacts, and optional samples
- `docs/`: taxonomy, summary, and interview-facing docs
- `models/`: local model cache and checkpoints, not committed
- `outputs/`: experiment outputs; only `outputs/final_summary/` is intended for GitHub
- `scripts/`: final closed-loop entrypoints
- `src/`: reusable package code
- `archive/`: legacy scripts/configs from earlier iterative stages
- `tests/`: lightweight smoke checks for repository layout

## Data Directories
- `data/sample/`: tiny redacted examples or placeholder docs only
- `data/annotation/`: Gold, silver, semi-supervised datasets and stats
- `data/processed/`: reproducible train/valid/test splits and processed tables
- `data/mlm_corpus/`: MLM training corpus and corpus stats

## Artifact Types
- Raw data:
  - `spam_email_data.log`
  - any large raw logs under `data/raw/`
  - keep local, do not commit
- Intermediate artifacts:
  - teacher predictions
  - checkpoints
  - split files
  - training logs
  - keep local, regenerate when needed
- Final artifacts:
  - `outputs/final_summary/final_closed_loop_results.csv`
  - `outputs/final_summary/final_closed_loop_summary.md`
  - `outputs/final_summary/final_interview_bullets.md`
  - safe to commit

## What Should Go To GitHub
Commit:
- `README.md`
- `RUNBOOK.md`
- `PROJECT_STRUCTURE.md`
- `requirements.txt`
- `.gitignore`
- `configs/`
- `docs/`
- `scripts/`
- `src/`
- `archive/`
- `tests/`
- `outputs/final_summary/`
- small JSON summaries if they help explain results

Do not commit:
- `models/`
- `spam_email_data.log`
- `data/raw/`
- large local caches
- large experiment checkpoints
- full prediction dumps
- notebook caches
- IDE files
- virtual environments

## Final Entry Scripts
The active public pipeline keeps only these scripts in `scripts/`:
- `download_model_from_modelscope.py`
- `build_mlm_corpus.py`
- `train_dapt_mlm.py`
- `run_multilingual_dapt_comparison.py`
- `predict_all_unlabeled.py`
- `predict_text_transformer.py`
- `build_consensus_silver.py`
- `build_semi_supervised_dataset.py`
- `run_semi_supervised_comparison.py`
- `summarize_final_closed_loop.py`

Legacy scripts remain available under `archive/scripts_legacy/` for reference but are not part of the recommended path. They include earlier baseline, annotation, and fusion utilities used during project iteration.
