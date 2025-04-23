import pandas as pd
import numpy as np
import os


from typing import Optional
from neuralprophet import NeuralProphet, set_log_level
set_log_level('ERROR')

from model import AbstractModel


class Neuralprophet(AbstractModel):
    def __init__(self, autoreg_lag: int = 0, exog_lag: Optional[int] = None, confidence_level: float = 0.9, ar_layers: Optional[list[int]] = []):
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1' ## This is needed due to https://github.com/suno-ai/bark/pull/619

        self.autoreg_lag = autoreg_lag
        self.exog_lag = exog_lag
        self.ar_layers = ar_layers

        confidence_level = confidence_level
        boundaries = round((1 - confidence_level) / 2, 2)
        # NeuralProphet only accepts quantiles value in between 0 and 1
        self.quantiles = [boundaries, confidence_level + boundaries]

    def fit(self, y: pd.DataFrame, X_exog: Optional[pd.DataFrame] = None):
        y = y.to_frame() # pandas.series to dataframe
        data = y.rename(columns={'price': 'y'}) # NeuralProphet needs a 'y' column
        data['ds'] = data.index # and a 'ds' column for time

        # NeuralProphet needs y and exog combined in one big dataframe
        if X_exog is not None:
          #X_exog = X_exog.to_frame() # pandas.series to dataframe
          self.exog_columns = X_exog.columns
          data = pd.merge(left=data, right=X_exog, left_index=True, right_index=True)

        self.data = data

    def predict(self, forecast_horizon: int, X_exog: Optional[pd.DataFrame] = None) -> np.ndarray:


        model = NeuralProphet(
            n_lags=self.autoreg_lag,
            n_forecasts=forecast_horizon,
            quantiles=self.quantiles,
            ar_layers=self.ar_layers,
            )

        model.set_plotting_backend('plotly')

        # Setup lagged_regressors
        if X_exog is not None:
          n_lags = self.exog_lag if self.exog_lag else 3
          for col in self.exog_columns:
            model.add_lagged_regressor(col, n_lags=n_lags)


        metrics = model.fit(self.data)
        self.model = model # new (We need to save the model for later plotting)

        # Make new dataframe. It is `forecast_horizon` larger
        df_future = model.make_future_dataframe(self.data, n_historic_predictions=True, periods=forecast_horizon)
        forecast = model.predict(df_future)

        # Prepering data to fit return structure
        prediction_ci = model.get_latest_forecast(forecast)
        prediction = prediction_ci[['ds', 'origin-0']]
        prediction.set_index('ds', inplace=True)

        ci_lower = prediction_ci[['ds', 'origin-0 5.0%']]
        ci_lower.set_index('ds', inplace=True)

        ci_upper = prediction_ci[['ds', 'origin-0 95.0%']]
        ci_upper.set_index('ds', inplace=True)

        ci = pd.concat([ci_upper, ci_lower], axis=1)
        ci.columns = ['upper', 'lower']

        return prediction, ci

    def plot(self):
        return self.model.plot(self.forecast)


if __name__ == '__main__':
    print("NeuralProphet constructed succesfully")
