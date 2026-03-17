# Interview Bullets

- Built an end-to-end mail risk classification pipeline from 24k unlabeled logs to production-style supervised + semi-supervised training assets.
- Replaced online model dependency with local ModelScope checkpoint download and local-path-only training/inference.
- Used domain-adaptive masked language modeling on the full unlabeled corpus before downstream 5-way classification.
- Designed a high-precision consensus silver strategy across linear models, tree models, transformer teachers, and rule signals.
- Kept evaluation clean by using Gold-only valid/test and training-only trusted silver with lower sample weights.
- Produced reproducible artifacts for data quality, DAPT, downstream comparison, trusted silver selection, semi-supervised comparison, and final reporting.
