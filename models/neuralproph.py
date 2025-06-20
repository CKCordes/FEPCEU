import pandas as pd
import numpy as np
import warnings
import os


from typing import Optional
from neuralprophet import NeuralProphet, set_log_level
set_log_level('ERROR')

from models.model import AbstractModel


class Neuralprophet(AbstractModel):
    def __init__(self, autoreg_lag: int = 48, 
                 exog_lag: int = 3, 
                 confidence_level: float = 0.9, 
                 ar_layers: Optional[list[int]] = [],
                 n_changepoints: int = 25,
                 daily_seasonality: bool = True,
                 weekly_seasonality: bool = True):
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1' # This is needed due to https://github.com/suno-ai/bark/pull/619
        self.forecaster = None
        self.autoreg_lag = autoreg_lag
        self.exog_lag = exog_lag
        self.ar_layers = ar_layers
        self.n_changepoints = n_changepoints
        self.daily_seasonality = daily_seasonality
        self.weekly_seasonality = weekly_seasonality

        confidence_level = confidence_level
        boundaries = round((1 - confidence_level) / 2, 2)

        # NeuralProphet only accepts quantiles value in between 0 and 1
        self.quantiles = [boundaries, confidence_level + boundaries]

    def fit(self, y: pd.DataFrame, X_exog: Optional[pd.DataFrame] = None):
        y = y.to_frame() 
        data = y.rename(columns={'price': 'y'}) # NeuralProphet needs a 'y' column
        data['ds'] = data.index # and a 'ds' column for time

        # Saves the cutoff day for later 
        self.cutoff_day = data['ds'].max()

        # NeuralProphet needs y and exog combined in one big dataframe
        if X_exog is not None:
          self.exog_columns = X_exog.columns
          data = pd.merge(left=data, right=X_exog, left_index=True, right_index=True)

        self.data = data

    def predict(self, forecast_horizon: int, X_exog: Optional[pd.DataFrame] = None) -> np.ndarray:

        if self.autoreg_lag > 0:
          model = NeuralProphet(
              n_lags=self.autoreg_lag,
              n_forecasts=forecast_horizon,
              quantiles=self.quantiles,
              ar_layers=self.ar_layers,
              daily_seasonality=self.daily_seasonality,
              weekly_seasonality=self.weekly_seasonality,
              n_changepoints=self.n_changepoints,
              )
        else:
          model = NeuralProphet(
              quantiles=self.quantiles
          )

        # Setup lagged_regressors
        if X_exog is not None:
          for col in self.exog_columns:
            model.add_lagged_regressor(col, n_lags = self.exog_lag, normalize='minmax')

        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          model.fit(self.data)
        self.forecaster = model

        with warnings.catch_warnings():
          warnings.simplefilter('ignore')
          df_future = model.make_future_dataframe(self.data, n_historic_predictions=True, periods=forecast_horizon)
          forecast = model.predict(df_future)

        # Prepering data to fit return structure
        if self.autoreg_lag > 0:
          prediction_ci = model.get_latest_forecast(forecast)
          prediction = prediction_ci[['ds', 'origin-0']]
          prediction.set_index('ds', inplace=True)

          ci_lower = prediction_ci[['ds', 'origin-0 5.0%']]
          ci_lower.set_index('ds', inplace=True)

          ci_upper = prediction_ci[['ds', 'origin-0 95.0%']]
          ci_upper.set_index('ds', inplace=True)

          ci = pd.concat([ci_upper, ci_lower], axis=1)
          ci.columns = ['upper', 'lower']
        else:
          # Use standard yhat columns
          future_forecast = forecast[forecast['ds'] > self.cutoff_day].copy()
          prediction = future_forecast[['ds', 'yhat1']]
          prediction.set_index('ds', inplace=True)
          ci = future_forecast[[f'yhat1 {int(self.quantiles[1] * 100)}.0%', f'yhat1 {int(self.quantiles[0] * 100)}.0%']]

          ci.columns = ['upper', 'lower']

        return prediction, ci
