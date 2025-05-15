import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

import glob as glob
import os as os


class Plotter:
    def __init__(self, directory):
        """
        Initialize the Plotter with the directory containing the CSV files.
        """
        self.directory = directory

    def load_csv_files(self, pattern):
        """
        Load CSV files matching the given pattern from the directory.
        """
        pattern = pattern + '*' # Meaning e.g. pca_kmeans* - where * indicates anything
        pattern = os.path.join(self.directory, pattern)
        file_paths = glob.glob(pattern)

        data_frames = [pd.read_csv(file) for file in file_paths]

        self.data = data_frames

        return data_frames
    
    def plot_metrics(self, metric, title):
        """
        Plot one line per model showing {metric}_mean over index `i`, with std as error bars.
        """
        # Store metric values per model across all DataFrames
        data = self.data
        models = {}

        for i, df in enumerate(data):
            for _, row in df.iterrows():
                model = row['model']
                feature_group = row['feature_group']
                mean_val = row[f'{metric}_mean']
                std_val = row[f'{metric}_std']

                feature_and_model = f'{feature_group}_{model}'

                if model not in models:
                    models[feature_and_model] = {'x': [], 'mean': [], 'std': []}

                models[feature_and_model]['x'].append(i)
                models[feature_and_model]['mean'].append(mean_val)
                models[feature_and_model]['std'].append(std_val)

        # Plot each model as a scatter line with error bars
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
            title=title,
            xaxis_title="Index (i)",
            yaxis_title=f"{metric.upper()}",
            template='plotly_white'
        )

        fig.show()


    def generate_plots(self):
        """
        Generate plots for MSE, MAE, and elapsed time.
        """
        # Load all relevant CSV files
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
