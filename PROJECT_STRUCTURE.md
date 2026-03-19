# PROJECT_STRUCTURE

## Top-Level Directories
- `README.md`: project homepage for GitHub and interviews
- `RUNBOOK.md`: reproducible execution guide
- `PROJECT_STRUCTURE.md`: repository layout and publishing policy
- `configs/`: active config files for baseline, Transformer, and LLM-guided runs
- `scripts/`: runnable entrypoints for the current public pipeline
- `src/`: reusable package code
- `docs/`: taxonomy, summary, and interview notes
- `data/`: local data artifacts, mostly private and ignored by Git
- `outputs/`: local experiment outputs; only `outputs/final_summary/` is public by default
- `archive/`: legacy iteration artifacts kept for reference
- `tests/`: lightweight repo smoke checks

## Active Script Surface
Current primary scripts:
- `download_model_from_modelscope.py`
- `build_mlm_corpus.py`
- `train_dapt_mlm.py`
- `run_multilingual_dapt_comparison.py`
- `label_with_llm.py`
- `train_llm_guided_transformer.py`
- `compare_llm_guided_training.py`
- `predict_all_unlabeled.py`
- `predict_text_transformer.py`
- `build_consensus_silver.py`
- `build_semi_supervised_dataset.py`
- `run_semi_supervised_comparison.py`
- `summarize_final_closed_loop.py`

Support script:
- `check_llm_guided_quality.py`

## Data Layout
- `data/sample/`: placeholder docs or tiny redacted examples only
- `data/annotation/`: Gold, silver, semi-supervised data tables and stats
- `data/processed/`: train/valid/test splits and processed tables
- `data/mlm_corpus/`: text corpus used for DAPT / MLM

## Output Layout
- `outputs/final_summary/`: safe public summary files
- `outputs/formal_baselines/`: local baseline artifacts
- `outputs/multilingual_dapt_comparison/`: local DAPT comparison artifacts
- `outputs/llm_guided/`: local LLM distillation experiments
- `outputs/llm_labeling/`: local LLM labeling outputs
- `outputs/semi_supervised_comparison/`: local semi-supervised outputs
- `outputs/predictions/`: teacher prediction dumps
- `outputs/dapt_multilingual_bert/`: local MLM training checkpoints

## What Goes To GitHub
Commit:
- root docs (`README.md`, `RUNBOOK.md`, `PROJECT_STRUCTURE.md`)
- `configs/`
- `scripts/`
- `src/`
- `docs/`
- `tests/`
- `outputs/final_summary/`
- small safe JSON summaries if they help explain results

Do not commit:
- `spam_email_data.log`
- `models/`
- raw or private annotation CSVs
- prediction dumps
- full checkpoints (`*.pt`, `*.bin`, `*.safetensors`)
- large intermediate outputs

## Public Storyline
This repository is organized around one engineering narrative:
1. build Gold from unlabeled enterprise mail logs
2. use unlabeled data for DAPT and teacher prediction
3. validate trusted silver carefully
4. test whether LLM-generated supervision helps or hurts a smaller model
5. keep only clean, reproducible artifacts in the public repo
