# Iris Data Lab — Assignment 1

A Streamlit data app that previews CSV uploads, draws interactive charts, and predicts a numeric target with a two-input linear regression.

## What the app does

1. Loads a bundled public Iris CSV, or lets the user upload another CSV.
2. Displays the first 20 rows and the dataset dimensions.
3. Lets the user select a numeric column and a line, scatter, or histogram chart. Scatter and line charts also offer an X-axis selector.
4. Lets the user select two different numeric inputs and a separate numeric target.
5. Fits a linear regression and displays a prediction for two typed values.
6. Includes a sentence explaining the model's purpose and the consequences of errors.

The bundled dataset has 150 rows, four numeric columns measured in centimeters, and a text species column. The default target is petal length. You can download the CSV from the app and upload it again to demonstrate the upload requirement.

## Run locally

Use Python 3.12. Open a terminal in this directory:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed in the terminal.

## Deploy through GitHub and Streamlit Community Cloud

1. Sign in to [GitHub](https://github.com/) and create a **public** repository named `assignment-1-streamlit`.
2. Extract the project ZIP and upload its **contents** into the repository. Keep `app.py`, `model.py`, and `requirements.txt` at the repository root, and keep `iris.csv` inside the `data` folder. Include `.streamlit/config.toml` if your upload method supports hidden folders; the app also works without that optional theme configuration. Do not upload `.venv`, passwords, or confidential data.
3. Commit the files to the `main` branch.
4. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and connect your GitHub account if prompted.
5. Choose **Create app**, then **Yup, I have an app** if prompted. Select the repository, branch `main`, and main file `app.py`.
6. Under **Advanced settings**, select Python **3.12** to match the tested environment, then deploy.
7. Open the resulting `https://….streamlit.app` URL. Confirm sharing is public, then open it in a private/incognito window while signed out to verify anyone can view it.
8. Download and upload `iris.csv`, try all three charts, choose the two inputs and target, and click **Predict**.

Official deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## What to submit

After deployment, submit these two **actual** URLs:

- Public Streamlit URL: copy it from your deployed app.
- Public GitHub repository URL: copy it from your repository page.

The project ZIP is a deployment package; a local URL or a file download does not replace either required submission URL.

## Understanding the regression

- The model learns `target = intercept + weight_1 × input_1 + weight_2 × input_2`.
- Only numeric columns are offered. Inputs and target must all be different.
- Missing or infinite values are removed only from the three selected columns.
- At least 10 usable rows are needed for a meaningful train/test demonstration. Constant columns and perfectly collinear training inputs are rejected with an explanation.
- An 80/20 random split with seed 42 measures held-out mean absolute error (MAE) and R². MAE has the same units as the target. R² can be negative and is undefined if all test targets are identical.
- After evaluation, the model is refitted on all usable rows for the interactive prediction. The test score estimates performance of the split model; it is not an independent evaluation of that final refit.
- The prediction row is a pandas DataFrame with exactly the same column names and order as training: `pd.DataFrame([values], columns=features)`.
- Predictions outside the observed input range are flagged. The model estimates associations, not causal effects.
- The public sample's numeric target is a flower measurement, not its species class.

The model-purpose sentence on the app is:

> This model estimates a selected numeric measurement from two other measurements for exploratory analysis; when it is wrong, it can produce misleading estimates and conclusions that should be checked against real observations.

## Verify locally

```bash
python -m unittest -v test_app.py
```

The tests exercise CSV loading, all three chart types, predictions, changes to the target, invalid uploads, missing values, and matching feature names/order.

## Dataset attribution

Fisher, R. A. (1936). *Iris* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76

Dataset page: https://archive.ics.uci.edu/dataset/53/iris

UCI lists the dataset under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/). The CSV was exported from scikit-learn's bundled `load_iris` dataset. Changes: readable column names ending in `_cm` and an added species-name column. No confidential or job-related data is included.

## Files

| File | Role |
| --- | --- |
| `app.py` | Streamlit page and widgets |
| `model.py` | Validation, model fitting, and prediction |
| `requirements.txt` | Tested dependency versions |
| `data/iris.csv` | Public sample data |
| `.streamlit/config.toml` | Optional theme and upload-size settings |
| `test_app.py` | Automated verification |
