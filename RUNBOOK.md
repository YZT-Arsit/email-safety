# RUNBOOK

This document is the full reproduction guide for the final public version of the repository.

## 1. Environment Setup
Command:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Input:
- repository source code

Output:
- local Python environment

If memory is tight:
- install dependencies first, then run one experiment at a time

## 2. Download Local Model From ModelScope
Command:
```bash
python scripts/download_model_from_modelscope.py \
  --model-id AI-ModelScope/bert-base-multilingual-cased \
  --cache-dir models
```
Input:
- ModelScope model id

Output:
- `models/bert-base-multilingual-cased/`
- `outputs/modelscope_download/download_summary.json`

If storage is tight:
- keep only the final local model directory and delete duplicate cache copies outside `models/bert-base-multilingual-cased/`

## 3. Build MLM Corpus
Command:
```bash
python scripts/build_mlm_corpus.py \
  --input-path spam_email_data.log \
  --output-path data/mlm_corpus/mail_corpus.txt \
  --stats-json data/mlm_corpus/mail_corpus_stats.json
```
Input:
- `spam_email_data.log`

Output:
- `data/mlm_corpus/mail_corpus.txt`
- `data/mlm_corpus/mail_corpus_stats.json`

If storage is tight:
- keep `mail_corpus_stats.json`; regenerate `mail_corpus.txt` when needed

## 4. Run DAPT / MLM Continued Pretraining
Command:
```bash
python scripts/train_dapt_mlm.py \
  --model-dir models/bert-base-multilingual-cased \
  --corpus-txt data/mlm_corpus/mail_corpus.txt \
  --output-dir outputs/dapt_multilingual_bert \
  --block-size 128 \
  --batch-size 8 \
  --epochs 2 \
  --learning-rate 5e-5
```
Input:
- local multilingual BERT checkpoint
- MLM corpus

Output:
- `outputs/dapt_multilingual_bert/checkpoint-epoch-*`
- `outputs/dapt_multilingual_bert/final_model/`
- `outputs/dapt_multilingual_bert/run_config.json`
- `outputs/dapt_multilingual_bert/trainer_state.json`
- `outputs/dapt_multilingual_bert/training_log.jsonl`

If GPU memory is tight:
- reduce `--batch-size` from `8` -> `4` -> `2`
- reduce `--block-size` from `128` -> `64`
- reduce `--epochs` to `1`

## 5. Run Gold v2 Downstream Comparison
Command:
```bash
python scripts/run_multilingual_dapt_comparison.py \
  --gold-csv data/annotation/gold/gold_v2.csv \
  --base-model-dir models/bert-base-multilingual-cased \
  --dapt-model-dir outputs/dapt_multilingual_bert/final_model \
  --processed-dir data/processed/multilingual_dapt_comparison \
  --output-dir outputs/multilingual_dapt_comparison \
  --max-length 256 \
  --batch-size 8 \
  --epochs 3 \
  --lr 2e-5
```
Input:
- `data/annotation/gold/gold_v2.csv`
- local base model dir
- local DAPT model dir

Output:
- `data/processed/multilingual_dapt_comparison/train.csv`
- `data/processed/multilingual_dapt_comparison/valid.csv`
- `data/processed/multilingual_dapt_comparison/test.csv`
- `outputs/multilingual_dapt_comparison/results_summary.csv`
- per-experiment metrics under `outputs/multilingual_dapt_comparison/*/`

If GPU memory is tight:
- reduce `--batch-size` from `8` -> `4`
- reduce `--max-length` from `256` -> `192` or `128`

## 6. Predict the Full 24k Unlabeled Set
Prerequisite:
- baseline teacher models under `outputs/formal_baselines/models/` already exist locally
- if not, regenerate them using the archived baseline scripts under `archive/scripts_legacy/`

Baseline teacher predictions:
```bash
python scripts/predict_all_unlabeled.py \
  --config configs/baseline.yaml \
  --model-path outputs/formal_baselines/models/text_only_lr/text_only_lr.joblib \
  --input-path spam_email_data.log \
  --raw-format log \
  --rules-config configs/weak_label_rules.yaml \
  --output-csv outputs/predictions/text_only_lr_all.csv
```

```bash
python scripts/predict_all_unlabeled.py \
  --config configs/baseline.yaml \
  --model-path outputs/formal_baselines/models/structured_only_lgbm/structured_only_lgbm.joblib \
  --input-path spam_email_data.log \
  --raw-format log \
  --rules-config configs/weak_label_rules.yaml \
  --output-csv outputs/predictions/structured_only_lgbm_all.csv
```

```bash
python scripts/predict_all_unlabeled.py \
  --config configs/baseline.yaml \
  --model-path outputs/formal_baselines/models/text_plus_structured_lr/text_plus_structured_lr.joblib \
  --input-path spam_email_data.log \
  --raw-format log \
  --rules-config configs/weak_label_rules.yaml \
  --output-csv outputs/predictions/text_plus_structured_lr_all.csv
```

Transformer teacher predictions:
```bash
python scripts/predict_text_transformer.py \
  --config configs/text_transformer.yaml \
  --checkpoint outputs/multilingual_dapt_comparison/multilingual_bert_base/best_model.pt \
  --input-path spam_email_data.log \
  --raw-format log \
  --output-csv outputs/predictions/multilingual_bert_base_all.csv
```

```bash
python scripts/predict_text_transformer.py \
  --config configs/text_transformer.yaml \
  --checkpoint outputs/multilingual_dapt_comparison/multilingual_bert_dapt/best_model.pt \
  --input-path spam_email_data.log \
  --raw-format log \
  --output-csv outputs/predictions/multilingual_bert_dapt_all.csv
```
Input:
- full unlabeled mail log
- saved baseline and Transformer checkpoints

Output:
- `outputs/predictions/*.csv`

If GPU memory is tight for Transformer prediction:
- reduce the batch size in `configs/text_transformer.yaml`
- reduce `model.max_length`

## 7. Build Consensus Trusted Silver
Command:
```bash
python scripts/build_consensus_silver.py \
  --text-lr-csv outputs/predictions/text_only_lr_all.csv \
  --structured-lgbm-csv outputs/predictions/structured_only_lgbm_all.csv \
  --fusion-lr-csv outputs/predictions/text_plus_structured_lr_all.csv \
  --mbert-csv outputs/predictions/multilingual_bert_base_all.csv \
  --dapt-mbert-csv outputs/predictions/multilingual_bert_dapt_all.csv \
  --output-csv data/annotation/silver/consensus_trusted_silver.csv \
  --stats-json data/annotation/silver/consensus_trusted_silver_stats.json \
  --min-teachers-agree 4
```
Input:
- five teacher prediction files

Output:
- `data/annotation/silver/consensus_trusted_silver.csv`
- `data/annotation/silver/consensus_trusted_silver_stats.json`

If silver count is too small:
- lower `--min-teachers-agree` carefully
- loosen class thresholds only after checking precision risk

## 8. Build Semi-Supervised Dataset
Command:
```bash
python scripts/build_semi_supervised_dataset.py \
  --gold-csv data/annotation/gold/gold_v2.csv \
  --silver-csv data/annotation/silver/consensus_trusted_silver.csv \
  --output-csv data/annotation/semi_supervised/semi_supervised_train.csv \
  --stats-json data/annotation/semi_supervised/semi_supervised_stats.json \
  --gold-weight 1.0 \
  --silver-weight 0.5
```
Input:
- Gold v2
- consensus trusted silver

Output:
- `data/annotation/semi_supervised/semi_supervised_train.csv`
- `data/annotation/semi_supervised/semi_supervised_stats.json`

If silver quality is uncertain:
- keep `--silver-weight` at `0.3` or `0.5`
- do not raise silver weight before reviewing per-class impact

## 9. Run Semi-Supervised Comparison
Command:
```bash
python scripts/run_semi_supervised_comparison.py \
  --gold-csv data/annotation/gold/gold_v2.csv \
  --silver-csv data/annotation/silver/consensus_trusted_silver.csv \
  --base-model-dir models/bert-base-multilingual-cased \
  --dapt-model-dir outputs/dapt_multilingual_bert/final_model \
  --processed-dir data/processed/semi_supervised_comparison \
  --output-dir outputs/semi_supervised_comparison \
  --max-length 256 \
  --batch-size 8 \
  --epochs 3 \
  --lr 2e-5 \
  --silver-weight 0.5
```
Input:
- Gold v2
- trusted silver
- local base and DAPT checkpoints

Output:
- `outputs/semi_supervised_comparison/results_summary.csv`
- per-experiment metrics under `outputs/semi_supervised_comparison/*/`

If GPU memory is tight:
- reduce `--batch-size`
- reduce `--max-length`

## 10. Generate Final Public Summary
Command:
```bash
python scripts/summarize_final_closed_loop.py \
  --gold-v1-csv data/annotation/clean_labeled_dataset.csv \
  --gold-v2-stats-json data/annotation/gold/gold_v2_stats.json \
  --mlm-stats-json data/mlm_corpus/mail_corpus_stats.json \
  --download-summary-json outputs/modelscope_download/download_summary.json \
  --baseline-summary-csv outputs/formal_baselines/results_summary.csv \
  --dapt-summary-csv outputs/multilingual_dapt_comparison/results_summary.csv \
  --consensus-silver-stats-json data/annotation/silver/consensus_trusted_silver_stats.json \
  --semi-summary-csv outputs/semi_supervised_comparison/results_summary.csv \
  --output-dir outputs/final_summary
```
Input:
- key stats and experiment outputs

Output:
- `outputs/final_summary/final_closed_loop_results.csv`
- `outputs/final_summary/final_closed_loop_summary.md`
- `outputs/final_summary/final_interview_bullets.md`
