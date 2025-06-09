import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from models.model import AbstractModel
import statsmodels.api as sm

from typing import Optional, List

warnings.filterwarnings("ignore", category=FutureWarning)

class Arima(AbstractModel):
    def __init__(self, order: List[int], seasonal_order: List[int]):
        self.forecaster = None
        self.results = None
        self.order = order
        self.seasonal_order = seasonal_order
    
    def fit(self, y: pd.DataFrame, X_exog: Optional[pd.DataFrame] = None):
        self.forecaster = sm.tsa.statespace.SARIMAX (
            y,
            exog=X_exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=False
        )
        print("Fitting ARIMA model with order:", self.order, "and seasonal order:", self.seasonal_order)
        self.results = self.forecaster.fit()
        print("Model fitted successfully.")


    def predict(self, forecast_horizon, X_exog: Optional[pd.DataFrame] = None) -> tuple:
        # Forecast
        print("Forecasting with ARIMA model...")
        forecast = self.results.get_forecast(steps=forecast_horizon, exog=(X_exog.iloc[:forecast_horizon] if X_exog is not None else None))
        print("Forecasting completed.")
        return pd.DataFrame(forecast.predicted_mean), pd.DataFrame(forecast.conf_int())