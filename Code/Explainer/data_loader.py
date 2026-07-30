from __future__ import annotations
from typing import Iterable, Optional, Tuple

import pandas as pd
import torch


def load_data(
    input_data_path: str,
    drop_columns: Optional[Iterable[str]] = None,
    expected_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, list[str]]:
    
    print(f"Loading data from: {input_data_path}")
    data = pd.read_csv(input_data_path)
    print(f"Raw CSV shape: {data.shape[0]} rows x {data.shape[1]} columns")

    if drop_columns:
        requested = list(drop_columns)
        missing = [column for column in requested if column not in data.columns]
        if missing:
            raise ValueError(f"Columns requested for removal were not found: {missing}")
        data = data.drop(columns=requested)
        print(f"Shape after dropping non-feature columns: {data.shape}")

    if expected_samples is not None and len(data) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, but the loaded feature data has {len(data)} rows."
        )

    nonnumeric = data.select_dtypes(exclude=["number"]).columns.tolist()
    if nonnumeric:
        raise TypeError(
            "All model-input columns must already be numeric. "
            f"Non-numeric columns: {nonnumeric}"
        )

    if data.isna().any().any():
        na_counts = data.isna().sum()
        na_counts = na_counts[na_counts > 0].to_dict()
        raise ValueError(f"Input features contain missing values: {na_counts}")

    feature_names = data.columns.astype(str).tolist()
    X = torch.as_tensor(data.to_numpy(dtype="float32"), dtype=torch.float32)
    print(f"Tensor shape passed to SHAP: {tuple(X.shape)}")
    return X, feature_names

