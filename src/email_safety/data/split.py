from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def make_train_valid_split(
    df: pd.DataFrame,
    label_column: str,
    valid_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[label_column] if stratify else None
    train_df, valid_df = train_test_split(
        df,
        test_size=valid_size,
        random_state=random_state,
        stratify=y,
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True)
