# Interview Notes

## 1-Minute Version
I built a closed-loop email risk classification project starting from 24k unlabeled mail logs. I first defined a 5-class risk taxonomy, then created Gold v1 and Gold v2 through staged annotation and targeted relabeling. On top of that I used the full unlabeled corpus in two ways: domain-adaptive pretraining with MLM on a local multilingual BERT, and high-precision trusted silver built from multi-teacher consensus. I compared traditional baselines, multilingual BERT, DAPT multilingual BERT, and semi-supervised settings while keeping validation and test strictly Gold-only.

## 3-Minute Version
The core challenge was that the raw dataset had no labels, mixed text and structured fields, and a realistic amount of noise. So I treated the problem as a full data-and-model closed loop rather than a pure classifier training task.

First, I defined a 5-class taxonomy for email risk and built a Gold dataset in two rounds, growing from 1000 to 1491 manually confirmed samples. I also built baseline models to establish lower-cost references and to generate teacher signals.

Second, I used all 24k unlabeled emails to do domain-adaptive pretraining with masked language modeling. Because the training environment could not rely on Hugging Face online downloads, I switched to ModelScope and made the whole pipeline load models only from local paths.

Third, I used five teacher signals plus rule information to build a conservative trusted silver set. Then I compared Gold-only and Gold-plus-silver training with multilingual BERT and DAPT multilingual BERT, while keeping validation and test purely on Gold data.

The project is strong for interview discussion because it covers taxonomy design, data engineering, offline reproducibility, representation learning, pseudo-label quality control, and honest evaluation design.

## High-Frequency Follow-Up Questions
### Why did you use DAPT?
- The corpus is domain-specific and noisy.
- DAPT lets the encoder absorb email-security language patterns before supervised classification.
- It is a principled way to use the full unlabeled corpus.

### Why keep valid/test Gold-only?
- To avoid optimistic evaluation from noisy pseudo-labels.
- Silver should help training, not redefine the benchmark.

### Why was trusted silver so small?
- I optimized for precision instead of coverage.
- In security tasks, noisy positives can do more harm than good.
- High-risk classes used stricter thresholds intentionally.

### Why did semi-supervised training not beat DAPT Gold-only in the current run?
- The current trusted silver set is conservative and class-skewed.
- It mostly adds easy head-class examples, so the gain on long-tail classes is limited.
- That result is still useful because it shows disciplined evaluation instead of claiming pseudo-label gains uncritically.

### What parts are most interview-relevant?
- turning unlabeled data into a full training loop
- offline reproducible model workflow
- multi-teacher consensus for pseudo-label quality control
- separating train-only silver from clean evaluation
