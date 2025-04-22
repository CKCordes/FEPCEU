from models.model import AbstractModel
from typing import Optional, Tuple

import pandas as pd
from sktime.forecasting.chronos import ChronosForecaster
from sktime.forecasting.base import ForecastingHorizon

class Chronos(AbstractModel):
    def __init__(self):
        self.forcaster = None
        self.preds = None
    
    def fit(
            self, 
            y: pd.DataFrame, 
            X_exog: Optional[pd.DataFrame] = None
        ):

        self.forecaster = ChronosForecaster("amazon/chronos-t5-tiny")  

        self.forecaster.fit(y)  
        
    def predict(
            self, 
            forecast_horizon: int, 
            X_exog: Optional[pd.DataFrame] = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        fh = ForecastingHorizon(X_exog.index[:forecast_horizon], is_relative=False)
        y_pred = self.forecaster.predict(fh)  
        return pd.DataFrame(y_pred), None