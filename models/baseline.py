from models.model import AbstractModel
import pandas as pd
from typing import Optional, Tuple
import numpy as np
from skforecast.model_selection import TimeSeriesFold
from skforecast.recursive import ForecasterEquivalentDate

class Baseline(AbstractModel):
    def __init__(self):
        self.forecaster = None


    def fit(
            self, 
            y: pd.DataFrame, 
            X_exog: Optional[pd.DataFrame] = None
        ):
        self.forecaster = ForecasterEquivalentDate(
                 offset    = pd.DateOffset(days=1),
                 n_offsets = 1
             )

        self.forecaster.fit(y=y)


    def predict(
            self, 
            forecast_horizon: int, 
            X_exog: Optional[pd.DataFrame] = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.forecaster is None:
            raise ValueError("Model must be fit before predictions can be made")
        
        predictions = self.forecaster.predict(
            steps=forecast_horizon,
        )
        return pd.DataFrame(predictions), None
