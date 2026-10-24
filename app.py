"""Assignment 1: upload, explore, and predict with a public CSV."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from model import fit_regression, numeric_columns, predict_row

st.set_page_config(page_title="Iris Data Lab", page_icon="🌸", layout="wide")
st.title("🌸 Iris Data Lab")
st.write("Upload a CSV, explore its patterns, and make a prediction with two inputs.")

SAMPLE_PATH = Path(__file__).parent / "data" / "iris.csv"
sample_bytes = SAMPLE_PATH.read_bytes()
with st.sidebar:
    st.header("Your dataset")
    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    st.caption("Use public, nonconfidential data with at least three numeric columns.")
    st.download_button("Download sample Iris CSV", sample_bytes, "iris.csv", "text/csv")
    st.markdown("[Dataset source and license](https://archive.ics.uci.edu/dataset/53/iris)")

raw = uploaded.getvalue() if uploaded is not None else sample_bytes
dataset_key = sha256(raw).hexdigest()[:12]
try:
    df = pd.read_csv(BytesIO(raw))
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
    st.error("Cannot read this CSV. Please upload a nonempty, comma-separated UTF-8 file with a header row.")
    st.stop()
if df.empty:
    st.error("This CSV has no data rows. Please upload a CSV with data below its header.")
    st.stop()

if uploaded is None:
    st.info("The public Iris sample is loaded. Upload your own CSV to replace it.")
else:
    st.success(f"Loaded {uploaded.name}")

columns = numeric_columns(df)
stats = st.columns(3)
stats[0].metric("Rows", f"{len(df):,}")
stats[1].metric("Columns", len(df.columns))
stats[2].metric("Numeric columns", len(columns))
st.subheader("1 · Preview your data")
st.dataframe(df.head(20), hide_index=True, width="stretch")
st.caption("Preview shows the first 20 rows. Charts and the model use the dataset, not just this preview.")
if len(columns) < 3:
    st.error("Please choose a CSV with at least three numeric columns for two inputs and a separate target.")
    st.stop()

st.subheader("2 · Explore a chart")
left, right = st.columns(2)
chart_type = left.selectbox("Chart type", ["Scatter", "Line", "Histogram"], key=f"chart_{dataset_key}")
column = right.selectbox("Column to chart (Y)", columns, index=columns.index("petal_length_cm") if "petal_length_cm" in columns else 0, key=f"column_{dataset_key}")
if chart_type == "Histogram":
    bins = st.slider("Number of bins", 5, 50, 20, key=f"bins_{dataset_key}")
    plot_data = pd.DataFrame({"value": df[column]}).replace([np.inf, -np.inf], np.nan).dropna()
    chart = alt.Chart(plot_data).mark_bar(color="#6C63CF").encode(
        x=alt.X("value:Q", bin=alt.Bin(maxbins=bins), title=column),
        y=alt.Y("count():Q", title="Number of rows"), tooltip=[alt.Tooltip("count():Q", title="Rows")],
    )
else:
    x_options = [c for c in columns if c != column]
    if chart_type == "Line":
        x_options = ["Row number"] + x_options
    x_column = st.selectbox("X-axis", x_options, key=f"x_{dataset_key}_{chart_type}_{column}")
    x_values = np.arange(1, len(df) + 1) if x_column == "Row number" else df[x_column]
    plot_data = pd.DataFrame({"x": x_values, "y": df[column]}).replace([np.inf, -np.inf], np.nan).dropna()
    base = alt.Chart(plot_data).encode(
        x=alt.X("x:Q", title=x_column), y=alt.Y("y:Q", title=column),
        tooltip=[alt.Tooltip("x:Q", title=x_column), alt.Tooltip("y:Q", title=column)],
    )
    chart = base.mark_circle(size=65, color="#6C63CF", opacity=0.75) if chart_type == "Scatter" else base.mark_line(color="#6C63CF", point=True)
    if chart_type == "Line":
        st.caption("The line connects observations in increasing X order; a row number is not a time measurement.")
if plot_data.empty:
    st.warning("No finite values are available for this chart. Choose another column.")
else:
    # Avoid Altair's row limit for large uploads while keeping small CSVs intact.
    if len(plot_data) > 5000:
        chart = chart.properties(data=plot_data.sample(5000, random_state=42))
        st.caption("Chart uses a reproducible sample of 5,000 rows for display.")
    st.altair_chart(chart.properties(height=340), width="stretch")

st.subheader("3 · Train a linear regression")
target = st.selectbox("Target to predict", columns, index=columns.index("petal_length_cm") if "petal_length_cm" in columns else 2, key=f"target_{dataset_key}")
available = [c for c in columns if c != target]
selectors = st.columns(2)
first = selectors[0].selectbox("Input 1", available, key=f"first_{dataset_key}_{target}")
second_options = [c for c in available if c != first]
second = selectors[1].selectbox("Input 2", second_options, key=f"second_{dataset_key}_{target}_{first}")
features = [first, second]
try:
    model, clean, metrics = fit_regression(df, features, target)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

st.caption(f"Using {len(clean):,} complete, finite rows; excluded {len(df) - len(clean):,} rows with missing or infinite values in the three selected columns.")
score_cols = st.columns(2)
score_cols[0].metric("Test mean absolute error", f"{metrics['mae']:.3f}")
score_cols[1].metric("Test R²", f"{metrics['r2']:.3f}" if metrics["r2"] is not None else "Not defined")
st.caption("Scores use an 80/20 train/test split (seed 42). MAE is in the target's units; lower is better. R² can be negative. The prediction model is then refitted on all usable rows.")

with st.expander("How the model works"):
    st.write("Linear regression learns an intercept and one weight for each selected input. It estimates a numeric value, not a flower species.")
    st.write(f"Predicted {target} = {model.intercept_:.4f} + ({model.coef_[0]:.4f} × {first}) + ({model.coef_[1]:.4f} × {second})")

st.subheader("4 · Try a prediction")
model_key = f"{dataset_key}_{target}_{first}_{second}"
with st.form(f"predict_{model_key}"):
    inputs = st.columns(2)
    values = []
    for index, feature in enumerate(features):
        values.append(inputs[index].number_input(
            f"Value for {feature}", value=float(clean[feature].median()),
            format="%.4f", key=f"value_{model_key}_{index}",
            help=f"Observed range: {clean[feature].min():g} to {clean[feature].max():g}",
        ))
    submitted = st.form_submit_button("Predict", type="primary")
if submitted:
    if not np.isfinite(values).all():
        st.error("Please enter two finite numbers.")
    else:
        prediction = predict_row(model, features, values)
        st.metric(f"Predicted {target}", f"{prediction:.3f}")
        if any(value < clean[feature].min() or value > clean[feature].max() for feature, value in zip(features, values)):
            st.warning("One or more inputs are outside the observed range, so this prediction may be less reliable.")

st.divider()
st.markdown("**Purpose and consequences:** This model estimates a selected numeric measurement from two other measurements for exploratory analysis; when it is wrong, it can produce misleading estimates and conclusions that should be checked against real observations.")
st.caption("Iris sample: R. A. Fisher (1936), UCI Machine Learning Repository, CC BY 4.0; exported from scikit-learn's bundled copy, with readable column names and species labels.")
