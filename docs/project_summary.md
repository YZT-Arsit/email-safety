# Project Summary

## Goal
Build an interview-grade enterprise email safety classifier from a realistic starting point: **24k unlabeled mail logs**, heterogeneous fields, noisy content, and no official training split.

## Main Technical Route
- define a 5-class email risk taxonomy
- create Gold v1 and Gold v2 through staged annotation and targeted relabeling
- establish traditional baselines on text and structured security features
- run DAPT / MLM on the full unlabeled corpus using a local multilingual BERT
- build trusted silver through multi-teacher consensus
- test LLM-to-BERT supervision with both reasoning hints and soft targets

## Key Experimental Findings
### Baselines
- best classic baseline: `text_only_lr`, macro F1 = `0.6325`
- structured-only model is weaker, confirming text is the main signal source

### DAPT
- `multilingual_bert_base`: `0.6045`
- `multilingual_bert_dapt`: `0.6088`
- conclusion: unlabeled corpus is useful for continued pretraining, but gains are modest and realistic

### Semi-Supervised Learning
- current trusted silver does not consistently improve the strongest DAPT backbone
- conclusion: silver quality control remains the main bottleneck, especially on long-tail labels

### LLM Distillation
- naive soft distillation with `alpha=0.5` hurts badly: `0.4964`
- reasoning hint alone is safer but still below plain BERT: `0.5622`
- best result comes from weak soft distillation with `alpha=0.1`: `0.6335`
- increasing epochs to `5` degrades performance again

## Closed-Loop Value
This project demonstrates more than a classifier:
- taxonomy design under weak supervision
- robust parsing and feature extraction on messy security logs
- disciplined use of unlabeled data through DAPT and teacher consensus
- honest evaluation of both positive and negative experimental outcomes
- an engineering mindset that prioritizes reproducibility over leaderboard-style overclaiming
