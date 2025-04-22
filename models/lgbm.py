import pandas as pd
import numpy as np
from datetime import datetime
from models.model import AbstractModel

from typing import Optional

from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import (
    TimeSeriesFold,
    OneStepAheadFold,
    bayesian_search_forecaster,
    backtesting_forecaster,
)
from skforecast.preprocessing import RollingFeatures


class LGBM(AbstractModel):
    def __init__(self):
        self.forecaster = None
        self.results = None
        self.cv_search = None
        
    def fit(
            self, 
            y: pd.DataFrame, # y_train
            X_exog: Optional[pd.DataFrame] = None
        ):
        window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
        
        # Hyperparameters search
        # ==============================================================================
        self.forecaster = ForecasterRecursive(
                        regressor        = LGBMRegressor(random_state=15926, verbose=-1),
                        lags             = 72,
                        window_features  = window_features,
                     )

        # Lags grid
        lags_grid = [48, 72, [1, 2, 3, 23, 24, 25, 167, 168, 169]]

        # Regressor hyperparameters search space
        def search_space(trial):
            search_space  = {
                'n_estimators'    : trial.suggest_int('n_estimators', 300, 1000, step=100),
                'max_depth'       : trial.suggest_int('max_depth', 3, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 25, 500),
                'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.5),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1),
                'max_bin'         : trial.suggest_int('max_bin', 50, 250),
                'reg_alpha'       : trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda'      : trial.suggest_float('reg_lambda', 0, 1),
                'lags'            : trial.suggest_categorical('lags', lags_grid)
            }
            return search_space

        # Folds training and validation
        print(f"len(y)={len(y)}")
        self.cv_search = TimeSeriesFold(steps = 36, initial_train_size = len(y)//10, fixed_train_size = False)


        results, best_trial = bayesian_search_forecaster(
            forecaster    = self.forecaster,
            y             = y, #df.loc[:end_validation, 'price'], # Test data not used
            exog          = X_exog, #df.loc[:end_validation, exog_features],
            cv            = self.cv_search,
            search_space  = search_space,
            metric        = 'mean_absolute_error',
            n_trials      = 20, # Increase this value for a more exhaustive search
            return_best   = True  # Forecaster object is updated with best config
        )

        self.forecaster.fit(
            y = y,
            exog = X_exog,
            store_in_sample_residuals = True
        )

        return self

    def predict(self, forecast_horizon, X_exog: Optional[pd.DataFrame] = None) -> tuple:
        if self.forecaster is None:
            raise ValueError("Model must be fit before predictions can be made")
        
        predictions = self.forecaster.predict(
            steps=forecast_horizon,
            exog=X_exog
        )

        return pd.DataFrame(predictions), None
