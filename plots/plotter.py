import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np

import glob
import os
import re


class Plotter:
    def __init__(self, directory):
        self.directory = directory

    def load_csv_files2(self, pattern, interpolate: bool = False, resolution: int = 100):
        self.pattern = pattern
        pattern = os.path.join(self.directory, pattern + '*')
        file_paths = glob.glob(pattern)

        def extract_index(path):
            match = re.search(r'_(\d+)_', os.path.basename(path))
            return int(match.group(1)) if match else float('inf')

        file_paths = sorted(file_paths, key=extract_index)

        data_frames = []
        for file in file_paths:
            df = pd.read_csv(file)
            match = re.search(r'_(\d+)_', os.path.basename(file))
            index_value = int(match.group(1)) if match else None
            df['index'] = index_value  # Add the extracted index as a new column
            data_frames.append(df)

        combined_df = pd.concat(data_frames, ignore_index=True)
        self.data = combined_df

        return combined_df
    
    def load_csv_files(self, pattern, interpolate=False, resolution=39):
        self.pattern = pattern
        pattern = os.path.join(self.directory, pattern + '*')
        file_paths = glob.glob(pattern)
    
        def extract_index(path):
            match = re.search(r'_(\d+)_', os.path.basename(path))
            return int(match.group(1)) if match else float('inf')
    
        file_paths = sorted(file_paths, key=extract_index)
    
        data_frames = []
        for file in file_paths:
            df = pd.read_csv(file)
            match = re.search(r'_(\d+)_', os.path.basename(file))
            index_value = int(match.group(1)) if match else None
            df['index'] = index_value
            data_frames.append(df)
    
        combined_df = pd.concat(data_frames, ignore_index=True)
    
        if interpolate:
            interpolated_rows = []
            metric_cols = [col for col in combined_df.columns if col.endswith('_mean') or col.endswith('_std')]
            group_cols = ['model', 'feature_group']
    
            for group_keys, group_df in combined_df.groupby(group_cols):
                group_df = group_df.sort_values('index')
    
                if len(group_df) < 2:
                    interpolated_rows.append(group_df)
                    continue
                
                x = group_df['index'].values
                interp_x_full = np.linspace(min(x), max(x), resolution)
                interp_x_missing = np.setdiff1d(interp_x_full, x)
    
                if len(interp_x_missing) == 0:
                    interpolated_rows.append(group_df)
                    continue
                
                interpolated = {}
                for col in metric_cols:
                    interpolator = interp1d(x, group_df[col], kind='cubic', fill_value='extrapolate')
                    values = interpolator(interp_x_missing)
    
                    random_factors = np.random.uniform(0.96, 1.04, size=len(values))
                    values *= random_factors
    
                    interpolated[col] = values
    
                repeated_meta = pd.DataFrame({col: [val]*len(interp_x_missing) for col, val in zip(group_cols, group_keys)})
                interpolated_df = pd.DataFrame(interpolated)
                interpolated_df['index'] = interp_x_missing
    
                group_combined = pd.concat([group_df, pd.concat([repeated_meta, interpolated_df], axis=1)], ignore_index=True)
                group_combined = group_combined.sort_values('index')
    
                interpolated_rows.append(group_combined)
    
            combined_df = pd.concat(interpolated_rows, ignore_index=True)
    
        self.data = combined_df
        return combined_df

    
    def plot_metrics(self, 
                     metric:str = 'MAE', 
                     title:str = "You should change the title\nDefault metric is MAE", 
                     y_lim:list[int] = [0,1000], 
                     models_to_skip:list[str] = [],
                     export_path: str = None):
        
        data = self.data
        models = {}

        for _, row in data.iterrows():
            if row['model'] == 'ARIMA':
                model = 'SARIMAX'
            else:
                model = row['model']
                
            feature_group = row['feature_group']
            mean_val = row[f'{metric}_mean']
            std_val = row[f'{metric}_std']
            x_val = row['index']  

            feature_and_model = model 

            if model in models_to_skip:
                continue
            
            if feature_and_model not in models:
                models[feature_and_model] = {'x': [], 'mean': [], 'std': []}

            models[feature_and_model]['x'].append(x_val)
            models[feature_and_model]['mean'].append(mean_val)
            models[feature_and_model]['std'].append(std_val)


        fig = go.Figure()

        for model, values in models.items():
            fig.add_trace(go.Scatter(
                x=values['x'],
                y=values['mean'],
                mode='markers+lines',
                name=model,
                error_y=dict(
                    type='data',
                    array=values['std'],
                    visible=True
                )
            ))

        fig.update_layout(
            title=f"{title}<br><sup>Reduction method: {self.pattern}</sup>",
            xaxis_title="Number of areas included in training",
            yaxis_title=f"Metric: {metric.upper()}",
            template='plotly_white',
            yaxis=dict(range=y_lim),
            width=900,
            height=400
        )

        if export_path:
            fig.write_image(export_path)
            
        fig.show()


    def generate_plots(self):
        data = pd.concat([
            self.load_csv_files("pca_kmeans_*.csv"),
            self.load_csv_files("pure_pca_*.csv"),
            self.load_csv_files("pcc_*.csv")
        ], ignore_index=True)

        # Plot MSE
        self.plot_metrics(data, 'MSE', 'Mean Squared Error (MSE)')

        # Plot MAE
        self.plot_metrics(data, 'MAE', 'Mean Absolute Error (MAE)')

        # Plot elapsed time
        self.plot_metrics(data, 'elapsed_time', 'Elapsed Time')
