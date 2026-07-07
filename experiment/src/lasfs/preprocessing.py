from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_cols, categorical_cols = infer_column_types(X)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def original_feature_mapping(preprocessor: ColumnTransformer) -> list[str]:
    mapping: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        cols = list(cols)
        if name == "num":
            mapping.extend(cols)
        elif name == "cat":
            onehot = transformer.named_steps["onehot"]
            for col, categories in zip(cols, onehot.categories_, strict=True):
                mapping.extend([col] * len(categories))
    return mapping

