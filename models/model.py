import pandas as pd
from typing import Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def fit(self, y: pd.DataFrame, X_exog: Optional[pd.DataFrame] = None):
        pass
    
    @abstractmethod
    def predict(self, forecast_horizon: int, X_exog: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
        