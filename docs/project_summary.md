# Project Summary

## Goal
Build a high-quality, interview-ready email risk classification project from a realistic starting point: 24k unlabeled mail logs with heterogeneous fields and no official labels.

## Main Route
- define a 5-class taxonomy for email risk
- create Gold v1 from seed annotation
- expand to Gold v2 through targeted relabeling
- establish strong traditional baselines
- continue pretraining a local multilingual BERT on the full unlabeled corpus with MLM
- compare base multilingual BERT vs DAPT multilingual BERT
- build high-precision trusted silver with multi-teacher consensus
- run semi-supervised comparison while keeping Gold-only validation and test sets

## Key Results
- Gold v1: `1000`
- Gold v2: `1491`
- MLM corpus retained rows: `23717 / 24000`
- Trusted silver: `127`
- Best baseline: `text_only_lr`, macro F1 = `0.6325`
- Best multilingual BERT: `multilingual_bert_dapt`, test macro F1 = `0.6088`
- Best semi-supervised result: `gold_v2_only__dapt_multilingual_bert`, test macro F1 = `0.6122`

## What the Closed Loop Demonstrates
- unlabeled security text can still be turned into a disciplined training pipeline
- DAPT is useful as a principled way to inject domain knowledge from raw logs
- trusted silver must be treated conservatively and should not contaminate evaluation
- reproducibility matters as much as raw modeling complexity in an interview setting
