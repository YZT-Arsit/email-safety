# Interview Notes

## 1-Minute Version
I built an enterprise email risk classification project starting from 24k unlabeled email logs. I defined a 5-class taxonomy, created Gold v1 and Gold v2 through targeted relabeling, and then used the unlabeled corpus in three ways: domain-adaptive pretraining, multi-teacher silver mining, and LLM-guided soft supervision. The strongest classic baseline reached macro F1 0.6325, DAPT multilingual BERT reached 0.6088, and the best LLM-guided BERT reached 0.6335 after I reduced distillation strength and prevented noisy soft labels from dominating training.

## 3-Minute Version
The hardest part of the project was not picking a model, but building a reliable learning loop from noisy, unlabeled enterprise email logs.

I started by defining a 5-class email risk taxonomy and building Gold data in two rounds, from 1000 to 1491 manually confirmed samples. In parallel, I engineered both text and structured security features, including URL statistics, attachment patterns, sender consistency, and network metadata.

Then I used the full 24k unlabeled corpus in multiple ways. First, I ran domain-adaptive pretraining with MLM on a local multilingual BERT because the runtime environment could not depend on Hugging Face online downloads. Second, I generated teacher predictions to build a conservative trusted silver set. Third, I used an LLM to produce class probabilities and reasoning traces, and tested whether those weak signals could improve a smaller Transformer through distillation.

The most interesting result was that LLM supervision was highly sensitive to distillation strength. A naive setting with strong soft-label alignment hurt performance badly, but reducing the distillation weight from 0.5 to 0.1 raised test macro F1 from 0.4964 to 0.6335, which surpassed both plain multilingual BERT and DAPT multilingual BERT. That let me show not just that I can add LLMs to a pipeline, but that I can diagnose when they help, when they hurt, and why.

## High-Frequency Follow-Up Questions
### Why did text-only logistic regression outperform some neural variants?
- The dataset is relatively small and class skewed.
- TF-IDF + LR is very strong on short security text and cheaper to serve.
- Strong baselines were necessary to avoid overstating Transformer gains.

### Why was DAPT useful but not dramatic?
- The unlabeled corpus is domain-specific, so DAPT helps the encoder absorb email security language.
- But 24k samples is still moderate-scale, so gains should be expected to be incremental, not huge.

### Why did trusted silver not consistently help?
- Silver is head-class heavy and still noisy on long-tail classes.
- Stronger backbones can overfit pseudo-label noise more quickly than simpler models.

### Why did `alpha=0.1` work better than `alpha=0.5` in distillation?
- The LLM signal had full coverage but only about 60% agreement with Gold labels.
- Large alpha over-trusted the noisy soft targets.
- Small alpha made the LLM act like a weak regularizer rather than a dominant teacher.

### Why did more epochs hurt the distillation runs?
- Longer training let the model absorb soft-label noise more aggressively.
- That is consistent with the agreement statistics and the observed class skew in LLM predictions.
