import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple, Union, Set
from sklearn.metrics import (
    root_mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    r2_score
)
import plotly.graph_objects as go
import re
import time
from models.model import AbstractModel
import shap
from datetime import datetime
import os

class EnhancedTimeSeriesExperiment:
    def __init__(self, 
                 models: Dict[str, AbstractModel], 
                 target_column: str,
                 metrics: Optional[Dict[str, Callable]] = None, 
                 test_size: float = 0.2,
                 forecast_horizon: int = 72,
                 n_splits: int = 5,
                 step_size: Optional[int] = None
        ):
        
        self.models = models
        self.target_column = target_column
        self.metrics = metrics or self._default_metrics()
        self.test_size = test_size
        self.forecast_horizon = forecast_horizon
        self.n_splits = n_splits
        self.step_size = step_size or forecast_horizon

        # Storage for different feature group experiments
        self.feature_group_results = {}
        self.feature_group_cv_results = {}
        self.feature_group_predictions = {}
        self.feature_group_cis = {}

    def _default_metrics(self) -> Dict[str, Callable]:
        return {
            'RMSE': root_mean_squared_error,
            'MAE': mean_absolute_error,
            'MAPE': mean_absolute_percentage_error,
            'R2': r2_score
        }
    
    def _identify_area_columns(self, df: pd.DataFrame) -> Dict[str, Dict[int, str]]:
        """
        Identify columns that represent different areas for different measurements.
        
        Returns:
            Dict with format: {measurement_type: {area_number: column_name}}
            Example: {'sun': {1: 'sun_area_1', 2: 'sun_area_2'}, 'wind': {1: 'wind_area_1'}}
        """
        area_columns = {}
        
        # Regular expression to match column names like "sun_area_1"
        pattern = r"^(\w+)_area_(\d+)$"
        
        for column in df.columns:
            match = re.match(pattern, column)
            if match:
                measurement_type = match.group(1)  # e.g., "sun", "wind", "temp"
                area_number = int(match.group(2))  # e.g., 1, 2, 3
                
                if measurement_type not in area_columns:
                    area_columns[measurement_type] = {}
                    
                area_columns[measurement_type][area_number] = column
        
        return area_columns
    
    def generate_cv_splits(
    self,
    df: pd.DataFrame,
    first_split_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> List[Tuple[int, int]]:
        """
        Generate indices for cross-validation splits using a sliding window approach.
        The training size remains constant across all splits, equal to the size of the first split.

        Args:
            df: DataFrame with datetime index
            first_split_date: Optional datetime (string or Timestamp) for the first train_end
                             If None, a default split will be calculated
        Returns:
            List of (train_start, train_end, test_end) indices
        """
        if isinstance(first_split_date, str):
            first_split_date = pd.Timestamp(first_split_date)

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")

        df_length = len(df)

        if first_split_date is not None:
            if first_split_date not in df.index:
                closest_date = df.index[df.index.get_indexer([first_split_date], method='nearest')[0]]
                print(f"Warning: {first_split_date} not found in index. Using closest date: {closest_date}")
                first_split_date = closest_date
            first_train_end = df.index.get_loc(first_split_date)
        else:
            first_train_end = int(df_length * 0.5)

        if first_train_end <= 0:
            raise ValueError("First train end must be positive")
        if first_train_end >= df_length - self.forecast_horizon:
            raise ValueError(f"First train end ({first_train_end}) too close to end of data. "
                            f"Need at least {self.forecast_horizon} points after it.")

        training_size = first_train_end 

        remaining_length = df_length - first_train_end
        max_splits = 1 + (remaining_length - self.forecast_horizon) // self.step_size

        actual_splits = min(self.n_splits, max_splits)
        if actual_splits < self.n_splits:
            print(f"Warning: Only {actual_splits} splits possible with current data and parameters (requested {self.n_splits})")

        # Generate splits
        splits = []
        for i in range(actual_splits):
            train_end = first_train_end + (i * self.step_size)
            train_start = train_end - training_size

            if train_start < 0:
                train_start = 0

            test_end = train_end + self.forecast_horizon

            if test_end > df_length:
                test_end = df_length

            splits.append((train_start, train_end, test_end))

        print("Cross-validation splits with sliding window:")
        for i, (train_start, train_end, test_end) in enumerate(splits):
            print(f"Split {i+1}: "
                  f"Train start = {df.index[train_start]} (index {train_start}), "
                  f"Train end = {df.index[train_end]} (index {train_end}), "
                  f"Test end = {df.index[test_end-1]} (index {test_end-1}), "
                  f"Training size = {train_end - train_start}")
            
        return splits
    
    def adjust_prices_with_wind(self, prices, exog_df, wind_scaling_factor=0.1, sun_scaling_factor=0.1, min_price_factor=0.5):
        

        wind_cols = []
        sun_cols = []

        pattern = re.compile(r"^(\w+)_area_(\d+)$")

        for col in exog_df.columns:
            match = pattern.match(col)
            if match:
                resource_type = match.group(1).lower()
                if resource_type == 'wind':
                    wind_cols.append(col)
                elif resource_type == 'sun':
                    sun_cols.append(col)
        if not wind_cols and not sun_cols:
            raise ValueError("No columns matching the pattern 'wind_area_X' or 'sun_area_Y' found")

        agg_wind = exog_df[wind_cols].mean(axis=1)
        wind_min = agg_wind.min()
        wind_max = agg_wind.max()
        if wind_max > wind_min:
            norm_wind = (agg_wind - wind_min) / (wind_max - wind_min)
        else:
            norm_wind = pd.Series(0, index=agg_wind.index)
        wind_corr = prices.corr(agg_wind)

        agg_sun = exog_df[sun_cols].mean(axis=1)
        sun_min = agg_sun.min()
        sun_max = agg_sun.max()
        if sun_max > sun_min:
            norm_sun = (agg_sun - sun_min) / (sun_max - sun_min)
        else:
            norm_sun = pd.Series(0, index=agg_sun.index)
        sun_corr = prices.corr(agg_sun)
        
        wind_effect = norm_wind * wind_scaling_factor
        sun_effect = norm_sun * sun_scaling_factor

        renewable_effect = (wind_effect + sun_effect).clip(upper=1.0)

        adjustment_factor = 1 - renewable_effect

        adjustment_factor = adjustment_factor.clip(lower=min_price_factor)

        adjusted_prices = prices * adjustment_factor
        
        adj_wind_corr = adjusted_prices.corr(agg_wind)
        
        adj_sun_corr = adjusted_prices.corr(agg_sun)
        
        combined_renewable = (norm_wind + norm_sun) / 2 if (wind_cols and sun_cols) else (norm_wind if wind_cols else norm_sun)
        orig_combined_corr = prices.corr(combined_renewable)
        adj_combined_corr = adjusted_prices.corr(combined_renewable)
        print("-" * 70)
        print("\nOverall Renewable Correlation:")
        print(f"Original correlation: {orig_combined_corr:.4f}")
        print(f"Adjusted correlation: {adj_combined_corr:.4f}")
        print(f"Change: {adj_combined_corr - orig_combined_corr:.4f}")
        print("-" * 70)
        print("\nWind Correlation:")
        print(f"Original correlation between prices and wind: {wind_corr:.4f}")
        print(f"Adjusted correlation between prices and wind: {adj_wind_corr:.4f}")
        print(f"Change: {adj_wind_corr - wind_corr:.4f}")
        print("-" * 70)
        print("\nSun Correlation:")
        print(f"Original correlation between prices and sun: {sun_corr:.4f}")
        print(f"Adjusted correlation between prices and sun: {adj_sun_corr:.4f}")
        print(f"Change: {adj_sun_corr - sun_corr:.4f}")
        print("-" * 70)

        return adjusted_prices
    
    def run_feature_group_experiments(
        self,
        df: pd.DataFrame,
        base_columns: Optional[List[str]] = None,
        area_config: Optional[Dict[str, List[Set[int]]]] = None,
        custom_feature_combinations: Optional[List[Dict[str, Set[int]]]] = None,
        first_split_date: Optional[Union[str, pd.Timestamp]] = None,
        add_all_columns: bool = True,
        add_base_columns: bool = True,
        manipulate: bool = False,
        manipulate_factor_wind: float = 0.1,
        manipulate_factor_sun: float = 0.1,
    ):
        """
        Run experiments with different combinations of area columns.
        
        Args:
            df: DataFrame with datetime index
            base_columns: List of basic columns to always include (non-area specific)
            area_config: Dict specifying which area combinations to test for each measurement type
                        Format: {measurement_type: [set(area_numbers), ...]}
                        Example: {'sun': [{1, 2}, {1, 2, 3}], 'wind': [{1}, {1, 2}]}
            custom_feature_combinations: List of specific measurement-area combinations to test
                                        Format: [{measurement_type: {area_numbers}, ...}, ...]
                                        Example: [{'sun': {1, 2, 3}, 'wind': {1}, 'temp': {1}}]
            first_split_date: Optional datetime for the first train_end
        """
        all_area_columns = self._identify_area_columns(df)
        print(f"Identified area columns: {all_area_columns}")
        
        if base_columns is None:
            base_columns = []
        
        feature_groups = []
        
        if add_base_columns:
            feature_groups.append({
                'name': 'base_only',
                'columns': base_columns
            })
        
        if area_config is not None:
            for measurement_type, area_sets in area_config.items():
                if measurement_type not in all_area_columns:
                    print(f"Warning: Measurement type '{measurement_type}' not found in data")
                    continue
                
                for area_set in area_sets:
                    combo_name = f"{measurement_type}_areas_" + "_".join(str(a) for a in sorted(area_set))
                    
                    valid_areas = [a for a in area_set if a in all_area_columns[measurement_type]]
                    combo_columns = [all_area_columns[measurement_type][a] for a in valid_areas]
                    
                    feature_groups.append({
                        'name': combo_name,
                        'columns': base_columns + combo_columns
                    })
        
        if custom_feature_combinations is not None:
            for idx, combo in enumerate(custom_feature_combinations):
                combo_name_parts = []
                combo_columns = list(base_columns) 
                
                for measurement_type, area_set in combo.items():
                    if measurement_type not in all_area_columns:
                        print(f"Warning: Measurement type '{measurement_type}' not found in data")
                        continue
                    
                    combo_name_parts.append(f"{measurement_type}:[{','.join(str(a) for a in sorted(area_set))}]")
                    
                    valid_areas = [a for a in area_set if a in all_area_columns[measurement_type]]
                    combo_columns.extend([all_area_columns[measurement_type][a] for a in valid_areas])
                
                combo_name = f"custom_area_{idx}"
                
                feature_groups.append({
                    'name': combo_name,
                    'columns': combo_columns
                })
        
        all_areas_columns = []
        for measurement_type, areas in all_area_columns.items():
            all_areas_columns.extend(areas.values())
        
        if add_all_columns:
            feature_groups.append({
                'name': 'all_areas',
                'columns': base_columns + all_areas_columns
            })
        
        cv_splits = self.generate_cv_splits(df, first_split_date)
        
        for feature_group in feature_groups:
            group_name = feature_group['name']
            exog_columns = feature_group['columns']
            
            print(f"\n{'='*80}\nRunning experiment for feature group: {group_name}")
            print(f"Using columns: {exog_columns}")
            
            self.feature_group_cv_results[group_name] = {}
            self.feature_group_predictions[group_name] = {}
            self.feature_group_cis[group_name] = {}
            
            for model_name in self.models.keys():
                self.feature_group_cv_results[group_name][model_name] = {metric: [] for metric in self.metrics.keys()}
                
                self.feature_group_cv_results[group_name][model_name]['elapsed_time'] = []
                self.feature_group_predictions[group_name][model_name] = []
                self.feature_group_cis[group_name][model_name] = []
                self.feature_group_cv_results[group_name][model_name]['SHAP_values'] = []

            for i, (train_start, train_end, test_end) in enumerate(cv_splits):
                print(f"Running CV split {i+1}/{len(cv_splits)} for feature group {group_name}")
                
                y_train = df.iloc[train_start:train_end][self.target_column]
                X_train = None if not exog_columns else df.iloc[train_start:train_end][exog_columns]
                
                y_test = df.iloc[train_end:test_end][self.target_column]
                X_test = None if not exog_columns else df.iloc[train_end:test_end][exog_columns]
                
                if len(y_test) == 0:
                    print(f"Warning: Test set for split {i+1} is empty. Skipping.")
                    continue
                
                # Run each model on this fold
                for model_name, model in self.models.items():
                    try:
                        model_copy = model
                    
                        start_time = time.perf_counter()

                        model_copy.fit(y_train, X_train)
                        
                        forecast_len = min(self.forecast_horizon, len(y_test))
                        predictions, ci = model_copy.predict(forecast_len, X_test)
                        
                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time

                        # Store predictions and confidence intervals
                        self.feature_group_predictions[group_name][model_name].append(predictions)
                        self.feature_group_cis[group_name][model_name].append(ci)
                        
                        y_true = y_test.iloc[:len(predictions)]
                        
                        if len(y_true) != len(predictions):
                            min_len = min(len(y_true), len(predictions))
                            y_true = y_true.iloc[:min_len]
                            predictions = predictions.iloc[:min_len]
                        
                        for metric_name, metric_func in self.metrics.items():
                            try:
                                metric_value = metric_func(y_true, predictions)
                                self.feature_group_cv_results[group_name][model_name][metric_name].append(metric_value)
                            except Exception as e:
                                print(f"Error calculating {metric_name} for {model_name} in split {i+1}: {str(e)}")
                                continue
                        self.feature_group_cv_results[group_name][model_name]['elapsed_time'].append(elapsed_time)
                        
                        if hasattr(model, 'compute_shap_values') and callable(getattr(model, 'compute_shap_values')):
                            self.feature_group_cv_results[group_name][model_name]['SHAP_values'].append(model.shap_values)
                    except Exception as e:
                        print(f"Error processing model {model_name} in split {i+1}: {str(e)}")
                        continue
            
            # Calculate aggregate statistics for this feature group
            self.feature_group_results[group_name] = {}
            for model_name, metrics in self.feature_group_cv_results[group_name].items():
                self.feature_group_results[group_name][model_name] = {}
                for metric_name, values in metrics.items():
                    if metric_name == "SHAP_values":
                        continue
                    if values: 
                        self.feature_group_results[group_name][model_name][f"{metric_name}_mean"] = np.mean(values)
                        self.feature_group_results[group_name][model_name][f"{metric_name}_std"] = np.std(values)

    def save_feature_group_results_to_csv(self, filepath: str):
        """
        Save the feature group evaluation results to a timestamped CSV file.
    
        This method:
        - Ensures the output directory exists.
        - Appends a timestamp to the provided filepath to prevent overwriting.
        - Converts the nested dictionary `self.feature_group_results` into a flat list of rows.
        - Writes the results to a CSV file using pandas.
        - Prints the final path of the saved file.
        
        Args:
            filepath (str): The base filepath (including directory and filename) where the CSV should be saved.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        base, ext = os.path.splitext(filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath_with_timestamp = f"{base}_{timestamp}{ext}"

        rows = []
        for group_name, models in self.feature_group_results.items():
            for model_name, metrics in models.items():
                row = {
                    'feature_group': group_name,
                    'model': model_name
                }
                row.update(metrics)
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath_with_timestamp, index=False)
        print(f"Saved feature group results to {filepath_with_timestamp}")

    
    def summarize_feature_group_results(self, metric: str = 'MAE') -> pd.DataFrame:
        """
        Summarize the results of feature group experiments for a specific metric.
        
        Returns:
            DataFrame with feature groups as rows and models as columns
        """
        results = {}
        
        for group_name, group_results in self.feature_group_results.items():
            results[group_name] = {}
            for model_name, model_metrics in group_results.items():
                metric_mean_key = f"{metric}_mean"
                metric_std_key = f"{metric}_std"
                
                if metric_mean_key in model_metrics and metric_std_key in model_metrics:
                    results[group_name][model_name] = f"{model_metrics[metric_mean_key]:.4f} ± {model_metrics[metric_std_key]:.4f}"
                    
                    results[group_name][f"{model_name}_raw"] = model_metrics[metric_mean_key]
                else:
                    results[group_name][model_name] = "N/A"
                    results[group_name][f"{model_name}_raw"] = float('inf')
        
        results_df = pd.DataFrame.from_dict(results, orient='index')
        
        if len(self.models) > 0:
            best_model = list(self.models.keys())[0]
            results_df = results_df.sort_values(f"{best_model}_raw")
            
        for model_name in self.models.keys():
            if f"{model_name}_raw" in results_df.columns:
                results_df = results_df.drop(columns=[f"{model_name}_raw"])
        
        return results_df
    
    def plot_feature_group_results(self, metric: str = 'MAE', top_n: int = None):
        """
        Plot the results of feature group experiments for a specific metric.
        
        Args:
            metric: The metric to plot (e.g., 'MAE', 'MSE')
            top_n: If provided, only plot the top N performing feature groups
        """
        feature_groups = list(self.feature_group_results.keys())
        models = list(self.models.keys())
        
        if metric == "SHAP_values":
            for group in feature_groups:
                for model_name in models:
                    shap_vals = self.feature_group_cv_results[group][model_name]['SHAP_values']
                    if shap_vals == []:
                        continue
                    print(f"Plotting SHAP summary plot for {model_name}")
                    for shap_val in shap_vals:
                        shap.summary_plot(shap_val, plot_type='bar')
            
            return
        plot_data = []
        for group_name in feature_groups:
            row = {'group': group_name}
            for model_name in models:
                model_metrics = self.feature_group_results[group_name].get(model_name, {})
                metric_mean = model_metrics.get(f"{metric}_mean")
                if metric_mean is not None:
                    row[model_name] = metric_mean
            plot_data.append(row)
        
        plot_df = pd.DataFrame(plot_data)
        
        if len(models) > 0 and len(plot_df) > 0:
            sort_col = models[0]
            if sort_col in plot_df.columns:
                plot_df = plot_df.sort_values(sort_col)
        
        if top_n is not None and len(plot_df) > top_n:
            plot_df = plot_df.head(top_n)
        
        ordered_groups = plot_df['group'].tolist() if len(plot_df) > 0 else feature_groups
        
        fig = go.Figure()
        
        for model_name in models:
            means = []
            errors = []
            
            for group_name in ordered_groups:
                model_metrics = self.feature_group_results[group_name].get(model_name, {})
                metric_mean = model_metrics.get(f"{metric}_mean")
                metric_std = model_metrics.get(f"{metric}_std")
                
                if metric_mean is not None and metric_std is not None:
                    means.append(metric_mean)
                    errors.append(metric_std)
                else:
                    means.append(None)
                    errors.append(None)
            
            fig.add_trace(go.Bar(
                name=model_name,
                x=ordered_groups,
                y=means,
                error_y=dict(
                    type='data',
                    array=errors,
                    visible=True
                )
            ))
        
        fig.update_layout(
            title=f"Feature Group Comparison - {metric}" + (f" (Top {top_n})" if top_n else ""),
            xaxis_title="Feature Group",
            yaxis_title=metric,
            width=1000,
            height=500,
            margin=dict(l=20, r=20, t=35, b=120),
            hovermode="closest",
            barmode='group'
        )
        fig.update_xaxes(tickangle=-45)
        fig.show()
        
    def plot_feature_importance(self, metric: str = 'MAE'):
        """
        Create a visualization that shows the importance of different measurement types and areas.
        """
        
        if 'base_only' not in self.feature_group_results:
            print("Baseline 'base_only' group not found. Cannot compute relative importance.")
            return
            
        models = list(self.models.keys())
        if not models:
            print("No models available for analysis.")
            return
            
        model_name = models[0]
        
        baseline_metrics = self.feature_group_results['base_only'].get(model_name, {})
        baseline_value = baseline_metrics.get(f"{metric}_mean")
        
        if baseline_value is None:
            print(f"Baseline metric {metric} not available for model {model_name}.")
            return
            
        area_columns = {}
        for group_name, group_results in self.feature_group_results.items():
            if group_name == 'base_only' or group_name == 'all_areas':
                continue
                
            if 'custom_' in group_name:
                continue
                
            parts = group_name.split('_areas_')
            if len(parts) != 2:
                continue
                
            measurement_type = parts[0]
            area_str = parts[1]
            
            group_metrics = group_results.get(model_name, {})
            group_value = group_metrics.get(f"{metric}_mean")
            
            if group_value is None:
                continue
                
            improvement = baseline_value - group_value
            
            if measurement_type not in area_columns:
                area_columns[measurement_type] = []
                
            area_columns[measurement_type].append({
                'areas': area_str,
                'improvement': improvement
            })
        
        for measurement_type, improvements in area_columns.items():
            if not improvements:
                continue
                
            improvements.sort(key=lambda x: x['improvement'], reverse=True)
            
            fig = go.Figure(go.Bar(
                x=[imp['areas'] for imp in improvements],
                y=[imp['improvement'] for imp in improvements],
                text=[f"{imp['improvement']:.4f}" for imp in improvements],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Improvement in {metric} with different {measurement_type} areas",
                xaxis_title="Areas included",
                yaxis_title=f"Improvement in {metric} (higher is better)",
                width=800,
                height=400
            )
            fig.show()
    
    def plot_feature_group_predictions(self, df: pd.DataFrame, feature_group: str, fold_index: int = 0):
        """
        Plot predictions for a specific feature group and fold.
        """
        for model_name in self.models.keys():
            if (
                feature_group not in self.feature_group_predictions or
                model_name not in self.feature_group_predictions[feature_group] or
                not self.feature_group_predictions[feature_group][model_name] or
                fold_index >= len(self.feature_group_predictions[feature_group][model_name])
            ):
                print(f"No predictions available for model {model_name} in feature group {feature_group}, fold {fold_index}")
                continue
            
            fold_predictions = self.feature_group_predictions[feature_group][model_name][fold_index]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                name='Prediction', 
                x=fold_predictions.index, 
                y=fold_predictions.iloc[:, 0],
                mode='lines'
            ))
            
            fig.add_trace(go.Scatter(
                name='Real value', 
                x=df.index, 
                y=df[self.target_column], 
                mode='lines'
            ))
            
            if (
                feature_group in self.feature_group_cis and
                model_name in self.feature_group_cis[feature_group] and
                len(self.feature_group_cis[feature_group][model_name]) > fold_index and
                self.feature_group_cis[feature_group][model_name][fold_index] is not None
            ):
                ci = self.feature_group_cis[feature_group][model_name][fold_index]
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
            
            fig.update_layout(
                title=f"{model_name}'s Prediction vs Real Values - {feature_group} (Fold {fold_index+1})",
                xaxis_title="Date",
                yaxis_title=self.target_column,
                width=800,
                height=400,
                margin=dict(l=20, r=20, t=35, b=20),
                hovermode="x",
                legend=dict(orientation="h", yanchor="top", y=1.1, xanchor="left", x=0.001),
            )
            fig.show()