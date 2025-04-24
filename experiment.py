import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple, Union
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score
)
import plotly.graph_objects as go

from models.model import AbstractModel

class TimeSeriesExperiment:
    def __init__(self, 
                 models: Dict[str, AbstractModel], 
                 target_column: str,
                 metrics: Optional[Dict[str, Callable]] = None, 
                 test_size: float = 0.2,
                 forecast_horizon: int = 72,
                 n_splits: int = 5,  # Number of validation splits
                 step_size: Optional[int] = None  # Step size between validation periods
        ):

        self.models = models
        self.target_column = target_column
        self.metrics = metrics or self._default_metrics()
        self.test_size = test_size
        self.forecast_horizon = forecast_horizon
        self.n_splits = n_splits
        self.step_size = step_size or forecast_horizon  # Default to forecast_horizon if not specified

        self.results = {}
        self.cv_results = {}  # Store cross-validation results
        self.predictions = {}
        self.cis = {}

    def _default_metrics(self) -> Dict[str, Callable]:
        return {
            'MSE': mean_squared_error,
            'MAE': mean_absolute_error,
            'MAPE': mean_absolute_percentage_error,
            'R2': r2_score
        }
    
    def generate_cv_splits(
        self, 
        df: pd.DataFrame, 
        first_split_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> List[Tuple[int, int]]:
        """
        Generate indices for cross-validation splits.
        
        Args:
            df: DataFrame with datetime index
            first_split_date: Optional datetime (string or Timestamp) for the first train_end
                             If None, a default split will be calculated
        
        Returns:
            List of (train_end, test_end) indices
        """
        # Convert first_split_date to pd.Timestamp if it's a string
        if isinstance(first_split_date, str):
            first_split_date = pd.Timestamp(first_split_date)
        
        # Check if df has a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
            
        df_length = len(df)
            
        # Calculate the start index for the first split
        if first_split_date is not None:
            # Find the index position of the date
            if first_split_date not in df.index:
                # Find the closest date that exists in the index
                closest_date = df.index[df.index.get_indexer([first_split_date], method='nearest')[0]]
                print(f"Warning: {first_split_date} not found in index. Using closest date: {closest_date}")
                first_split_date = closest_date
                
            # Get the index position
            first_train_end = df.index.get_loc(first_split_date)
        else:
            # Use default: 50% of data for first training split
            first_train_end = int(df_length * 0.5)
        
        # Ensure first_train_end is valid
        if first_train_end <= 0:
            raise ValueError("First train end must be positive")
            
        if first_train_end >= df_length - self.forecast_horizon:
            raise ValueError(f"First train end ({first_train_end}) too close to end of data. " 
                             f"Need at least {self.forecast_horizon} points after it.")
        
        # Calculate how many splits we can make
        remaining_length = df_length - first_train_end
        max_splits = 1 + (remaining_length - self.forecast_horizon) // self.step_size
        
        # Use the minimum of requested splits and maximum possible splits
        actual_splits = min(self.n_splits, max_splits)
        
        if actual_splits < self.n_splits:
            print(f"Warning: Only {actual_splits} splits possible with current data and parameters (requested {self.n_splits})")
        
        # Generate splits
        splits = []
        for i in range(actual_splits):
            # Calculate end of training set
            train_end = first_train_end + (i * self.step_size)
            
            # End of test set is forecast_horizon steps after train_end
            test_end = train_end + self.forecast_horizon
            
            # Ensure test_end doesn't exceed the dataframe length
            if test_end > df_length:
                test_end = df_length
            
            splits.append((train_end, test_end))
        
        # Print the splits with their corresponding dates for clarity
        print("Cross-validation splits:")
        for i, (train_end, test_end) in enumerate(splits):
            print(f"Split {i+1}: Train end = {df.index[train_end]} (index {train_end}), "
                  f"Test end = {df.index[test_end-1]} (index {test_end-1})")
            
        return splits
    
    def run_experiment(
        self, 
        df: pd.DataFrame, 
        exog_columns: Optional[List[str]] = None,
        first_split_date: Optional[Union[str, pd.Timestamp]] = None
    ):
        """
        Run time series experiment with cross-validation.
        
        Args:
            df: DataFrame with datetime index containing target and feature columns
            exog_columns: List of exogenous variable column names
            first_split_date: Optional datetime for the first train_end
        """
        # Initialize CV results storage
        for model_name in self.models.keys():
            self.cv_results[model_name] = {metric: [] for metric in self.metrics.keys()}
            self.predictions[model_name] = []
            self.cis[model_name] = []
        
        # Generate cross-validation splits
        cv_splits = self.generate_cv_splits(df, first_split_date)
        
        # Run each model through all CV splits
        for i, (train_end, test_end) in enumerate(cv_splits):
            print(f"Running CV split {i+1}/{len(cv_splits)}: "
                  f"Train end={df.index[train_end]}, Test end={df.index[test_end-1]}")
            
            # Get training data
            y_train = df.iloc[:train_end][self.target_column]
            X_train = None if exog_columns is None else df.iloc[:train_end][exog_columns]
            
            # Get test data (only the forecast_horizon steps after train_end)
            y_test = df.iloc[train_end:test_end][self.target_column]
            X_test = None if exog_columns is None else df.iloc[train_end:test_end][exog_columns]
            
            # Ensure y_test is not empty
            if len(y_test) == 0:
                print(f"Warning: Test set for split {i+1} is empty. Skipping.")
                continue
            
            # Run each model on this fold
            for model_name, model in self.models.items():
                try:
                    # Make a copy of the model to avoid state spillover between folds
                    # (This assumes models can be copied, you might need a different approach if not)
                    model_copy = model
                    
                    # Fit model
                    model_copy.fit(y_train, X_train)
                    
                    # Predict forecast_horizon steps
                    forecast_len = min(self.forecast_horizon, len(y_test))
                    predictions, ci = model_copy.predict(forecast_len, X_test)
                    
                    # Ensure we have predictions
                    if len(predictions) == 0:
                        print(f"Warning: No predictions generated for {model_name} in split {i+1}. Skipping.")
                        continue
                    
                    # Store predictions and confidence intervals
                    self.predictions[model_name].append(predictions)
                    if ci is not None:
                        self.cis[model_name].append(ci)
                    else:
                        self.cis[model_name].append(None)
                    
                    # Compare predictions against actual values
                    y_true = y_test.iloc[:len(predictions)]
                    
                    # Ensure y_true has same number of samples as predictions
                    if len(y_true) != len(predictions):
                        print(f"Warning: Number of test samples ({len(y_true)}) doesn't match predictions ({len(predictions)}). Adjusting.")
                        min_len = min(len(y_true), len(predictions))
                        y_true = y_true.iloc[:min_len]
                        predictions = predictions.iloc[:min_len]
                    
                    # Ensure we have data to evaluate
                    if len(y_true) == 0 or len(predictions) == 0:
                        print(f"Warning: No data to evaluate for {model_name} in split {i+1}. Skipping.")
                        continue
                    
                    # Evaluate metrics
                    for metric_name, metric_func in self.metrics.items():
                        try:
                            metric_value = metric_func(y_true, predictions)
                            self.cv_results[model_name][metric_name].append(metric_value)
                        except Exception as e:
                            print(f"Error calculating {metric_name} for {model_name} in split {i+1}: {str(e)}")
                            continue
                
                except Exception as e:
                    print(f"Error processing model {model_name} in split {i+1}: {str(e)}")
                    continue
        
        # Calculate aggregate statistics
        self.results = {}
        for model_name, metrics in self.cv_results.items():
            self.results[model_name] = {}
            for metric_name, values in metrics.items():
                if values:  # Only calculate if we have values
                    self.results[model_name][f"{metric_name}_mean"] = np.mean(values)
                    self.results[model_name][f"{metric_name}_std"] = np.std(values)
    
    def summarize_results(self) -> pd.DataFrame:
        """Return a DataFrame summarizing the cross-validation results"""
        return pd.DataFrame.from_dict(self.results, orient='index')
    
    def plot_predictions(self, df: pd.DataFrame, fold_index: Optional[int] = None):
        """
        Plot predictions vs actual values. 
        If fold_index is provided, only that fold is plotted. Otherwise, all folds are plotted.
        """
        for model_name in self.predictions.keys():
            if not self.predictions[model_name]:
                print(f"No predictions available for model {model_name}")
                continue
                
            if fold_index is not None:
                # Check if the fold index is valid
                if fold_index < 0 or fold_index >= len(self.predictions[model_name]):
                    print(f"Invalid fold index {fold_index} for model {model_name}")
                    continue
                
                folds_to_plot = [fold_index]
            else:
                folds_to_plot = range(len(self.predictions[model_name]))
            
            for fold in folds_to_plot:
                # Get predictions for this fold
                fold_predictions = self.predictions[model_name][fold]
                
                # Create figure
                fig = go.Figure()
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    name='Prediction', 
                    x=fold_predictions.index, 
                    y=fold_predictions.iloc[:, 0],
                    mode='lines'
                ))
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    name='Real value', 
                    x=df.index, 
                    y=df[self.target_column], 
                    mode='lines'
                ))
                
                # Add confidence intervals if available
                if len(self.cis[model_name]) > fold and self.cis[model_name][fold] is not None:
                    ci = self.cis[model_name][fold]
                    if len(ci) > 0:
                        fig.add_trace(go.Scatter(
                            name='Upper Bound', 
                            x=fold_predictions.index[:len(ci)], 
                            y=ci.iloc[:, 0], 
                            mode='lines',
                            marker=dict(color="#444"), 
                            line=dict(width=0), 
                            showlegend=False
                        ))
                        fig.add_trace(go.Scatter(
                            name='Lower Bound', 
                            x=fold_predictions.index[:len(ci)], 
                            y=ci.iloc[:, 1], 
                            marker=dict(color="#444"),
                            line=dict(width=0), 
                            mode='lines', 
                            fillcolor='rgba(68, 68, 68, 0.3)', 
                            fill='tonexty', 
                            showlegend=False
                        ))
                
                # Update layout
                fig.update_layout(
                    title=f"{model_name}'s Prediction vs Real Values (Fold {fold+1})",
                    xaxis_title="Date",
                    yaxis_title=self.target_column,
                    width=800,
                    height=400,
                    margin=dict(l=20, r=20, t=35, b=20),
                    hovermode="x",
                    legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0.001),
                )
                fig.show()
    
    def plot_cv_results(self, metric: str = 'MAE'):
        """
        Plot cross-validation results for a specific metric across all models.
        """
        fig = go.Figure()
        
        for model_name, metrics in self.cv_results.items():
            if metric in metrics and metrics[metric]:
                values = metrics[metric]
                
                # Add bar for each model with error bars
                fig.add_trace(go.Bar(
                    name=model_name,
                    y=[np.mean(values)],
                    x=[model_name],
                    error_y=dict(
                        type='data',
                        array=[np.std(values)],
                        visible=True
                    )
                ))
        
        fig.update_layout(
            title=f"Cross-Validation {metric} Results",
            xaxis_title="Model",
            yaxis_title=metric,
            width=800,
            height=400,
            margin=dict(l=20, r=20, t=35, b=20),
            hovermode="closest",
        )
        fig.show()
