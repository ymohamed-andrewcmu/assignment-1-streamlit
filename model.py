"""Small, reusable data-validation and regression functions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def numeric_columns(df):
    """Exclude nonnumeric and boolean fields from chart/model choices."""
    return [c for c in df.select_dtypes(include="number").columns if not pd.api.types.is_bool_dtype(df[c])]


def fit_regression(df, features, target):
    if len(features) != 2 or len(set(features + [target])) != 3:
        raise ValueError("Select two different inputs and a separate target.")
    if not all(c in numeric_columns(df) for c in features + [target]):
        raise ValueError("Inputs and target must be numeric columns.")
    clean = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 10:
        raise ValueError("At least 10 complete, finite rows are required to train and evaluate this model.")
    if any(clean[c].nunique() < 2 for c in features + [target]):
        raise ValueError("Each selected column must contain at least two distinct values.")
    X = clean[features]
    y = clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if np.linalg.matrix_rank(np.column_stack([np.ones(len(X_train)), X_train])) < 3:
        raise ValueError("The selected inputs are constant or perfectly related in the training data. Choose different inputs.")
    model = LinearRegression().fit(X_train, y_train)
    test_predictions = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, test_predictions)),
        "r2": float(r2_score(y_test, test_predictions)) if y_test.nunique() > 1 else None,
    }
    model.fit(X, y)
    return model, clean, metrics


def predict_row(model, features, values):
    # Preserve training names AND order, as required by the assignment.
    row = pd.DataFrame([values], columns=features)
    return float(model.predict(row)[0])
