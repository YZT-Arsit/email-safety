# RUNBOOK

This document is the reproducible command reference for the current public version of EmailSafety.

## 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Download Local Backbone From ModelScope
Input:
- ModelScope model id

Command:
```bash
python scripts/download_model_from_modelscope.py \
  --model-id AI-ModelScope/bert-base-multilingual-cased \
  --cache-dir models
```

Output:
- `models/bert-base-multilingual-cased/`
- `outputs/modelscope_download/download_summary.json`

If memory or storage is tight:
- keep only the final local model dir
- avoid duplicate cache copies

## 3. Build MLM Corpus
Input:
- `spam_email_data.log`

Command:
```bash
python scripts/build_mlm_corpus.py \
  --input-path spam_email_data.log \
  --output-path data/mlm_corpus/mail_corpus.txt \
  --stats-json data/mlm_corpus/mail_corpus_stats.json
```

Output:
- `data/mlm_corpus/mail_corpus.txt`
- `data/mlm_corpus/mail_corpus_stats.json`

## 4. Run DAPT / MLM
Input:
- local multilingual BERT
- MLM corpus

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

Output:
- `outputs/dapt_multilingual_bert/final_model/`
- `outputs/dapt_multilingual_bert/run_config.json`
- `outputs/dapt_multilingual_bert/trainer_state.json`
- `outputs/dapt_multilingual_bert/training_log.jsonl`

Low-memory settings:
- `--batch-size 8 -> 4 -> 2`
- `--block-size 128 -> 64`
- `--epochs 2 -> 1`

## 5. Gold v2 Downstream Comparison
Input:
- `data/annotation/gold/gold_v2.csv`
- local base and DAPT checkpoints

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

Output:
- `outputs/multilingual_dapt_comparison/results_summary.csv`
- per-run metrics and checkpoints under `outputs/multilingual_dapt_comparison/`

Low-memory settings:
- `--batch-size 8 -> 4`
- `--max-length 256 -> 192 or 128`

## 6. Automatic LLM Labeling
Input:
- Gold CSV or unlabeled log file
- LLM API key

Command:
```bash
export DEEPSEEK_API_KEY="YOUR_API_KEY"

python scripts/label_with_llm.py \
  --input-path data/annotation/gold/gold_v2.csv \
  --raw-format csv \
  --api-url https://api.deepseek.com/v1/chat/completions \
  --model deepseek-chat \
  --max-workers 16 \
  --silver-output outputs/llm_labeling/gold_v2_llm_silver.jsonl \
  --hard-output outputs/llm_labeling/gold_v2_llm_hard.jsonl \
  --summary-output outputs/llm_labeling/gold_v2_llm_summary.json
```

Output:
- `outputs/llm_labeling/*.jsonl`
- `outputs/llm_labeling/gold_v2_llm_summary.json`

If API cost or rate limit is tight:
- lower `--max-workers`
- run a small `--limit` first

## 7. Train LLM-Guided BERT
Input:
- `gold_v2_llm_guided.csv`
- local multilingual BERT

Plain baseline:
```bash
python scripts/train_llm_guided_transformer.py \
  --config configs/llm_guided_transformer.yaml \
  --output-dir outputs/llm_guided/plain_bert \
  --no-use-risk-hint \
  --no-use-soft-targets
```

Best current soft-distill setting:
```bash
python scripts/train_llm_guided_transformer.py \
  --config configs/llm_guided_transformer.yaml \
  --output-dir outputs/llm_guided/soft_only_alpha01 \
  --no-use-risk-hint \
  --use-soft-targets
```

Output:
- `outputs/llm_guided/*/results_summary.csv`
- `outputs/llm_guided/*/metrics_summary.json`
- `outputs/llm_guided/*/best_model.pt`

Low-memory settings:
- reduce `train.batch_size`
- reduce `model.max_length`

## 8. Teacher Predictions on Full Unlabeled Set
Input:
- teacher checkpoints
- `spam_email_data.log`

Commands:
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
python scripts/predict_text_transformer.py \
  --config configs/text_transformer.yaml \
  --checkpoint outputs/multilingual_dapt_comparison/multilingual_bert_dapt/best_model.pt \
  --input-path spam_email_data.log \
  --raw-format log \
  --output-csv outputs/predictions/multilingual_bert_dapt_all.csv
```

Output:
- `outputs/predictions/*.csv`

## 9. Build Consensus Trusted Silver
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

## 10. Build Semi-Supervised Dataset
```bash
python scripts/build_semi_supervised_dataset.py \
  --gold-csv data/annotation/gold/gold_v2.csv \
  --silver-csv data/annotation/silver/consensus_trusted_silver.csv \
  --output-csv data/annotation/semi_supervised/semi_supervised_train.csv \
  --stats-json data/annotation/semi_supervised/semi_supervised_stats.json \
  --gold-weight 1.0 \
  --silver-weight 0.5
```

## 11. Run Semi-Supervised Comparison
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

## 12. Summarize Final Results
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

## 13. Key Public Files
Keep for GitHub:
- `README.md`
- `RUNBOOK.md`
- `PROJECT_STRUCTURE.md`
- `docs/*`
- `outputs/final_summary/*`

Keep local only:
- raw logs
- annotation CSVs
- model checkpoints
- prediction dumps
- large experiment outputs
