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
        self.model = None
        self.results = None
        self.order = order
        self.seasonal_order = seasonal_order
    
    def fit(self, y: pd.DataFrame, X_exog: Optional[pd.DataFrame] = None):
        self.model = sm.tsa.statespace.SARIMAX (
            y,
            exog=X_exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=False
        )
        self.results = self.model.fit()


    def predict(self, forecast_horizon, X_exog: Optional[pd.DataFrame] = None) -> tuple:
        # Forecast
        forecast = self.results.get_forecast(steps=forecast_horizon, exog=(X_exog.iloc[:forecast_horizon] if X_exog is not None else None))
        return pd.DataFrame(forecast.predicted_mean), pd.DataFrame(forecast.conf_int())



"""
# TODO: If order is given in config, use that order and not auto_arima.
# https://alkaline-ml.com/pmdarima/tips_and_tricks.html?highlight=predict
class Arima(Model):
    def _train(self) -> None:
        if "order" in self.config:
            
            Assumes order contains a array-like with ARIMA order (shape=(3,))
            (p,d,q)
            if "seasonal_order" in self.config:
                Assumes array-like shape=(4,)
                (P,D,Q,s)
                
                self.model = pm.arima.ARIMA(order=self.config["order"], seasonal_order=self.config["seasonal_order"])
                print("Fitting seasonal ARIMA")
            else:
                self.model = pm.arima.ARIMA(order=self.config["order"])
                print("Fitting ARIMA")
            self.model.fit(self.train_data, X=self.train_exog)

        elif "pipeline" in self.config and self.config["pipeline"] == True:
            # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.pipeline.Pipeline.html#pmdarima.pipeline.Pipeline
            self.model = Pipeline([
                ("fourier", FourierFeaturizer(m=24)),
                ("arima", pm.auto_arima(y=self.train_data, X=self.train_exog, seasonal=False, m=24, trace=self.debug, maxiter=7))
            ])
            self.model.fit(self.train_data, X=self.train_exog)
            
        else:
            self.model = pm.auto_arima(y=self.train_data, X=self.train_exog, seasonal=True, m=24, trace=self.debug, maxiter=7)
            print("Fitting AutoARIMA")
            if self.debug:
                print("Model summary:")
                print(self.model.summary())

    def predict(self, start_date: datetime, hours_ahead: int):
        self.train_data, self.test_data, self.train_exog, self.test_exog = self.split_data(start_date, hours_ahead)
        self._train()
        self.prediction, self.ci = self.model.predict(n_periods=hours_ahead, X=self.test_exog, return_conf_int=True)
        return self.prediction, self.ci"
"""