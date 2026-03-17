# Final Closed Loop Summary

## Data Scale
- Gold v1: 1000
- Gold v2: 1491
- MLM corpus retained rows: 23717
- Consensus trusted silver: 127

## Model Download / DAPT
- ModelScope model id: AI-ModelScope/bert-base-multilingual-cased
- Local model dir: models/bert-base-multilingual-cased
- DAPT corpus file: data/mlm_corpus/mail_corpus.txt

## Best Results
- Best baseline: text_only_lr / macro_f1=0.6325
- Best multilingual BERT: multilingual_bert_dapt / test_macro_f1=0.6088
- Best semi-supervised: gold_v2_only__dapt_multilingual_bert / test_macro_f1=0.6122

## Closed Loop
- 2.4w unlabeled mails contributed in two ways: domain-adaptive MLM pretraining and high-precision consensus trusted silver.
- Gold stayed the only source for valid/test; silver was used only in train and with lower sample weight.

## Limitations
- Long-tail classes remain harder than the head class black_industry_or_policy_violation.
- Border cases between phishing / impersonation and advertisement / black_industry_or_policy_violation remain noisy.
- Trusted silver prioritizes precision, so recall and coverage are intentionally conservative.
