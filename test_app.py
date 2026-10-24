"""Run with: python -m unittest -v test_app.py"""

from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from model import fit_regression, predict_row

ROOT = Path(__file__).parent


class RegressionTests(unittest.TestCase):
    def test_known_relationship_and_column_order(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"second": rng.normal(size=80), "first": rng.normal(size=80)})
        df["target"] = 5 + 2 * df["first"] - 3 * df["second"]
        model, _, scores = fit_regression(df, ["first", "second"], "target")
        self.assertEqual(list(model.feature_names_in_), ["first", "second"])
        self.assertAlmostEqual(predict_row(model, ["first", "second"], [4, 2]), 7)
        self.assertLess(scores["mae"], 1e-10)

    def test_missing_and_infinite_rows_are_excluded(self):
        df = pd.read_csv(ROOT / "data/iris.csv")
        df.loc[0, "sepal_length_cm"] = np.inf
        df.loc[1, "petal_length_cm"] = np.nan
        _, clean, _ = fit_regression(df, ["sepal_length_cm", "sepal_width_cm"], "petal_length_cm")
        self.assertEqual(len(clean), 148)

    def test_target_leakage_and_constant_columns_are_rejected(self):
        df = pd.read_csv(ROOT / "data/iris.csv")
        with self.assertRaises(ValueError):
            fit_regression(df, ["sepal_length_cm", "petal_length_cm"], "petal_length_cm")
        df["sepal_width_cm"] = 1
        with self.assertRaises(ValueError):
            fit_regression(df, ["sepal_length_cm", "sepal_width_cm"], "petal_length_cm")


class AppTests(unittest.TestCase):
    def load_app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_sample_all_charts_and_prediction(self):
        app = self.load_app()
        chart = next(s for s in app.selectbox if s.label == "Chart type")
        for kind in ["Line", "Histogram", "Scatter"]:
            app.selectbox(key=chart.key).select(kind).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("vega_lite_chart")), 1)
        app.number_input[0].set_value(6)
        app.number_input[1].set_value(3)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(m.label == "Predicted petal_length_cm" for m in app.metric))

    def test_uploaded_csv_replaces_sample(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame({"ad_spend": rng.uniform(0, 100, 40), "visitors": rng.uniform(10, 200, 40)})
        df["sales"] = 3 * df.ad_spend + 2 * df.visitors
        uploaded = BytesIO(df.to_csv(index=False).encode())
        uploaded.name = "public_demo.csv"
        with patch("streamlit.file_uploader", return_value=uploaded):
            app = self.load_app()
            self.assertEqual(app.dataframe[0].value.shape, (20, 3))
            self.assertEqual(app.metric[0].value, "40")
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any(m.label == "Predicted sales" for m in app.metric))

    def test_invalid_uploads_show_useful_messages(self):
        for payload in [b"", b"name,value\na,1\n", b"a,b,c\n1,2,3\n"]:
            with self.subTest(payload=payload):
                uploaded = BytesIO(payload)
                uploaded.name = "invalid.csv"
                with patch("streamlit.file_uploader", return_value=uploaded):
                    app = self.load_app()
                    self.assertTrue(len(app.error) or len(app.warning))

    def test_changed_target_uses_distinct_inputs(self):
        app = self.load_app()
        target = next(s for s in app.selectbox if s.label == "Target to predict")
        target.select("sepal_width_cm").run()
        self.assertEqual(len(app.exception), 0)
        selection = {s.label: s.value for s in app.selectbox}
        self.assertEqual(len({selection[k] for k in ["Target to predict", "Input 1", "Input 2"]}), 3)


if __name__ == "__main__":
    unittest.main()
