from models.model import AbstractModel
from typing import Optional, Tuple

import pandas as pd

from nixtla import NixtlaClient

class TimeGPT(AbstractModel):
    def __init__(self):
        self.forcaster = None
        self.preds = None
        # Please dont scrape this
        self.API_KEY="nixak-dYY7rbjTvFo81RJJ8piW8yH7TWzZ5Ey0dnzleZE0O6pDkEBIEFd93iA5kBWK4QcgOL1eR3Asfnji6Zzj"


    def fit(
            self, 
            y: pd.DataFrame, 
            X_exog: Optional[pd.DataFrame] = None
        ):
        self.nixtla_client = NixtlaClient(api_key=self.API_KEY)
        self.train_df = pd.concat([y, X_exog], axis=1)
        
        # Stuff that TimeGPT wants...
        self.train_df.rename(columns={'price':'y'}, inplace=True)
        self.train_df['ds'] = self.train_df.index
        self.train_df['unique_id'] = 0
        
    def predict(
            self, 
            forecast_horizon: int, 
            X_exog: Optional[pd.DataFrame] = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        predict_df = self.nixtla_client.forecast(
            df=self.train_df,
            hist_exog_list = self.train_df.drop(columns=['y']).columns.tolist(), 
            h=forecast_horizon,
            level=[90],  # Generate a 90% confidence interval
            finetune_steps=60,  # Specify the number of steps for fine-tuning
            finetune_loss="mae",  # Use the MAE as the loss function for fine-tuning
            model="timegpt-1-long-horizon",  # Use the model for long-horizon forecasting
            time_col="ds",
            target_col="y",
            id_col='unique_id'
        )

        predict_df['ds'] = pd.to_datetime(predict_df['ds'])
        predict_df = predict_df.set_index('ds')

        # First DataFrame: predictions
        df_preds = predict_df[['TimeGPT']].rename(columns={'TimeGPT': 'preds'})

        # Second DataFrame: prediction intervals
        df_bounds = predict_df[['TimeGPT-hi-90', 'TimeGPT-lo-90']].rename(columns={
            'TimeGPT-hi-90': 'upper price',
            'TimeGPT-lo-90': 'lower price'
        })
        return df_preds, df_bounds

        