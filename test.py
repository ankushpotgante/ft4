import unittest
import joblib
import pandas as pd
from sklearn import metrics

class TestCaliforniaHousingModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("artifacts/model.joblib")
        data = pd.read_csv("data/california_housing_test.csv")

        cls.X = data[[
            "longitude",
            "latitude",
            "housing_median_age",
            "total_rooms",
            "total_bedrooms",
            "population",
            "households",
            "median_income"
        ]]
        cls.y = data["median_house_value"]

        cls.pred = cls.model.predict(cls.X)

    def test_prediction_length(self):
        # Predictions length should match number of labels
        self.assertEqual(len(self.pred), len(self.y))

    def test_mae_in_range(self):
        # Mean Absolute Error should be non-negative
        mae = metrics.mean_absolute_error(self.y, self.pred)
        self.assertGreaterEqual(mae, 0.0)

    def test_rmse_reasonable(self):
        # RMSE should not be astronomically large (sanity check)
        rmse = metrics.mean_squared_error(self.y, self.pred, squared=False)
        print("Model RMSE on test set:", round(rmse, 2))
        self.assertLess(rmse, 150000)  # Adjust threshold for your model

    def test_predictions_are_numeric(self):
        # Predictions should be numeric floats
        self.assertTrue(all(isinstance(p, (float, int)) for p in self.pred))
