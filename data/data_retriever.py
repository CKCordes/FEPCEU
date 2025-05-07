import pandas as pd
import requests
from functools import reduce
import datetime
import os
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Dataretreiver():
    def __init__(self, data_src:str = 'stormglass', debug=False, fill_missing=True, start_date: str = "2024-01-01", end_date: str = "2024-12-31", reduce:str = None, normalize:bool = False):
        """
        ARGS:
        """        
        self.debug = debug

        self.start_date = start_date
        self.end_date = end_date
        self.sun_df = None
        self.wind_df = None 
        self.temp_df = None
        self.reduction_type = reduce if reduce is not None else 'None'
        
        self.elspot_df = self._request_energinet('Elspotprices',fill_missing)
        
        if data_src == 'dmi':
            # This is by far the safest way to store API keys.
            self.DMI_API_KEY = '65201ca0-3e61-4550-b23d-faf0aa6f3857'
            self.sun_df = self._request_DMI("sun_last1h_glob", "sun", fill_missing)
            self.wind_df = self._request_DMI("wind_speed_past1h", "wind", fill_missing)
            self.temp_df = self._request_DMI("temp_mean_past1h", "temp", fill_missing)
        
        elif data_src == 'stormglass':
            path_prefix = os.path.dirname(os.path.realpath(__file__))
            self.sun_df = self._read_weather_csv_data(os.path.join(path_prefix, 'uv_unfolded.csv'), 'sun')
            self.wind_df = self._read_weather_csv_data(os.path.join(path_prefix, 'vind_unfolded.csv'), 'wind')
            self.temp_df = self._read_weather_csv_data(os.path.join(path_prefix, 'temp_unfolded.csv'), 'temp')
        
        else:
            raise ValueError("Invalid weather data source")
        
        if reduce == 'pca_kmeans':
            self.sun_df = self.pca_kmeans_reduction(df=self.sun_df, verbose=debug)
            self.wind_df = self.pca_kmeans_reduction(df=self.wind_df, verbose=debug)
            self.temp_df = self.pca_kmeans_reduction(df=self.temp_df, verbose=debug)

        elif reduce == 'pca_pure':
            self.sun_df = self.pca_pure_reduction(df=self.sun_df, verbose=debug)
            self.wind_df = self.pca_pure_reduction(df=self.wind_df, verbose=debug)
            self.temp_df = self.pca_pure_reduction(df=self.temp_df, verbose=debug)
            
        elif reduce == 'pearson':
            self.sun_df = self.correlation_reduce(df=self.sun_df, verbose=debug)
            self.wind_df = self.correlation_reduce(df=self.wind_df, verbose=debug)
            self.temp_df = self.correlation_reduce(df=self.temp_df, verbose=debug)
            
        elif reduce is not None:
            print('A reduce keyword was given but not recognized. Use either \'pca\' or \'pearson\'')

        self.combined = self._combine_dfs()

        if normalize:
            scaler = MinMaxScaler()
            self.combined = pd.DataFrame(
                scaler.fit_transform(self.combined),
                index = self.combined.index,
                columns = self.combined.columns
            )

        

    def _request_DMI(self, parameterId, parameterName, fill_missing):
        def request(offset, parameterId, limit=1000):
            response = requests.get(
                url='https://dmigw.govcloud.dk/v2/metObs/collections/observation/items?&api-key=%s' % (self.DMI_API_KEY),
                params={
                    "parameterId": f"{parameterId}",
                    "bbox": '7.87,54.58,11.2,57.82',
                    "datetime": f'{self.start_date}T00:00:00Z/{self.end_date}T23:00:00Z',
                    "limit": str(limit),
                    "offset": str(offset),
                }
            )
            return response

        datetime_start = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        datetime_end = datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        total_entries = (datetime_end - datetime_start).days * 24#total_entries = 366 * 24 + 365*24  # Number of data points
        offset = 0
        all_obs = []  # To collect all the observations

        response = request(offset, parameterId, total_entries)
        result = response.json()
        all_obs = result.get('features', [])

        # Now process the observations into a DataFrame
        obs_cleaned = [{
            "HourDK": feature["properties"]["observed"],
            f"{parameterName}": feature["properties"]["value"],
        } for feature in all_obs]

        obs_df = pd.DataFrame.from_dict(obs_cleaned)
        obs_df["HourDK"] = pd.to_datetime(obs_df["HourDK"])
        obs_df['HourDK'] = obs_df['HourDK'].dt.tz_convert(None)
        obs_df.set_index("HourDK", inplace=True)
        obs_df.sort_index(inplace=True, ascending=True)
        obs_df = obs_df.loc[~obs_df.index.duplicated(keep='first')]
        
        if fill_missing:
            return self._fill_missing_values(obs_df)
        else:
            return obs_df

    def _request_energinet(self, dataset, fill_missing, filter=True):
        limit = 0
        url = None
        if filter is True:
            url = 'https://api.energidataservice.dk/dataset/%s?start=%sT00:00&end=%sT23:00&limit=%s&filter={"PriceArea":["DK1"]}' % (dataset, self.start_date, self.end_date, limit)
        else: 
            url = 'https://api.energidataservice.dk/dataset/%s?start=%sT00:00&end=%sT23:00&limit=%s' % (dataset, self.start_date, self.end_date, limit)

        response = requests.get(url=url)

        result = response.json()
        records = result.get('records', [])

        df = pd.DataFrame.from_dict(records)
        if dataset == 'Elspotprices':
            df["date"] = pd.to_datetime(df["HourDK"])
            # Drop unnecassary columns
            df.drop(columns=['HourUTC', 'SpotPriceEUR', 'PriceArea', "HourDK"], inplace=True)
            df.rename(columns={"SpotPriceDKK": "price"}, inplace=True)

        elif dataset == 'GasDailyBalancingPrice':
            df["date"] = pd.to_datetime(df["GasDay"])
            df.drop(columns=['GasDay', 'EEXWithinDayEUR_MWh', 'SalesPriceDKK_kWh', 'MarginalPurchasePriceDKK_kWh', 'THEPriceDKK_kWh', 'MarginalSalePriceDKK_kWh', 'EEXLowestPriceSaleDKK_kWh', 'EEXHighestPricePurchaseDKK_kWh', 'NeutralGasPriceDKK_kWh', 'ExchangeRateEUR_DKK', 'EEXSpotIndexEUR_MWh'], inplace=True)            
            df.rename(columns={"PurchasePriceDKK_kWh": "gas_purchase_price"}, inplace=True)


        elif dataset == 'CO2Emis':
            df["date"] = pd.to_datetime(df["Minutes5DK"])
            df.drop(columns=['Minutes5UTC', 'Minutes5DK', 'PriceArea'], inplace=True)

        elif dataset == 'ElectricityProdex5MinRealtime':
            df["date"] = pd.to_datetime(df["Minutes5DK"])
            df.drop(columns=['Minutes5UTC', 'Minutes5DK', 'PriceArea', 'BornholmSE4', 'ExchangeGreatBelt', 'ExchangeGermany', 'ExchangeNetherlands', 'ExchangeGreatBritain', 'ExchangeNorway', 'ExchangeSweden'], inplace=True)
        
        df.set_index("date", inplace=True)
        df = df.loc[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True)
        df = df.resample('H').ffill()
        df.fillna(method='ffill', inplace=True)
        if fill_missing:
            return self._fill_missing_values(df)
        else:
            return df
        
    def _fill_missing_values(self, df):
        start_date = df.index.min().replace(hour=0, minute=0, second=0)  # Start of the first day
        end_date = df.index.max().replace(hour=23, minute=0, second=0)  # End of the last day
        complete_date_range = pd.date_range(start=start_date, end=end_date, freq='H')

        # Drop duplicates:
        df = df[~df.index.duplicated()]
        # Reindex your DataFrame to include all hours, filling missing values with the previous value
        df_complete = df.reindex(complete_date_range, method='ffill')

        # Check if there are any missing hours
        missing_hours = len(complete_date_range) - len(df.index)
        if self.debug:
            print("Filling missing values")
            print(f"Start date: {start_date}")
            print(f"End date: {end_date}")
            print(f"Total expected hours: {len(complete_date_range)}")
            print(f"Hours in original data: {len(df.index)}")
            print(f"Missing hours filled: {missing_hours}")

        # Verify there are exactly 8760 hours (or 8784 for leap years)
        expected_hours = 8760
        if pd.Timestamp(start_date).is_leap_year:
            expected_hours = 8784
        if self.debug:
            print(f"Hours in complete data: {len(df_complete)}")
            print(f"Expected hours for the year: {expected_hours}")

        return df_complete

    def _read_weather_csv_data(self, csv_path: str, col_name:str):
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.tz_localize(None) # Remove timezone
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        df = df.resample('H').ffill()

        df = df.loc[~df.index.duplicated(keep='first')]
        # Rename columns so we actually can combine them
        df.rename(columns=lambda x: col_name + '_' + x.lower(), inplace=True)
        return df

    def _combine_dfs(self):
        dataframes = [self.sun_df, self.wind_df, self.temp_df, self.elspot_df]
        return reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'), dataframes)
    
    def correlation_reduce(self, df:pd.DataFrame, num_of_cols: int=10, verbose:bool = False):
        # Check if the two dataframes are the same length
        cp_elspot_df = self.elspot_df.copy()

        if len(self.elspot_df.index) != len(df.index):
            print('WARNING: While correlation reducing, the provided data amount does not match the elspot length. Truncating')
            # Use the amount with the least amount of indexes
            max_index = min(df.index[-1], self.elspot_df.index[-1])
            df = df[:max_index]
            cp_elspot_df = self.elspot_df[:max_index]

        correlations = {}
        for col in df.columns:
            corr, _ = pearsonr(df[col], cp_elspot_df['price'])
            correlations[col] = corr
        # Step 2: Sort by absolute correlation strength
        sorted_cols = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        # Step 3: Select top N wind points
        top_columns = [col for col, corr in sorted_cols[:num_of_cols]]
        if verbose:
            print(f"Top {num_of_cols} most correlated with elspot price:")
            for col in top_columns:
                print(f"{col}: correlation = {correlations[col]:.4f}")
        return df[top_columns]
    
    def pca_kmeans_reduction(self, df:pd.DataFrame, num_of_cols:int = 10, verbose:bool = False):
        # Step 1: Standardize the features across time (important before PCA!)
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns
        )

        # Step 2: Apply PCA
        pca = PCA(n_components=num_of_cols)
        pca_features = pca.fit_transform(data_scaled.T)
        
        if verbose:
            inertias = []
            k_range = range(1, 21)

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=0).fit(pca_features)
                inertias.append(kmeans.inertia_)

            # Plot
            plt.plot(k_range, inertias, marker='o')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('Inertia (SSE)')
            plt.title(f'Elbow Method For Optimal k - {df.columns[1]}')
            plt.grid(True)
            plt.show()
        # Step 3: Cluster the PCA-transformed points
        kmeans = KMeans(n_clusters=num_of_cols, random_state=42)
        labels = kmeans.fit_predict(pca_features)

        # Step 4: For each cluster, select representative point (closest to cluster center)
        representatives = []
        for cluster_id in range(num_of_cols):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_points = pca_features[cluster_indices]
            center = kmeans.cluster_centers_[cluster_id]
            closest_idx = cluster_indices[np.argmin(np.linalg.norm(cluster_points - center, axis=1))]
            representatives.append(df.columns[closest_idx])

        if verbose:
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.2%} total variance retained")
            print("\nSelected representative wind points:")
            print(representatives)
            # Optional: Plot clustering if you want to visualize
            plt.figure(figsize=(8, 6))
            for label in np.unique(labels):
                idx = labels == label
                plt.scatter(pca_features[idx, 0], pca_features[idx, 1], label=f'Cluster {label}')
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black', marker='x', label='Centers')
            plt.title('Wind points clustered in PCA space')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend()
            plt.show()
        
        # Step 5: Create reduced dataset with selected points
        return df[representatives]

    def pca_pure_reduction(self, df:pd.DataFrame, num_of_cols:int = 10, verbose:bool = False):
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(df),
            index=df.index,
            columns=df.columns
        )

        # Step 1: Fit PCA on transposed data
        pca = PCA(n_components=data_scaled.shape[1])  # Choose enough components
        pca.fit(data_scaled)

        # Step 2: Get absolute loadings (features × components)
        loadings = np.abs(pca.components_.T)

        # Step 3: Sum absolute loadings aczross top k components (e.g., k=5 or all available)
        k = min(10, loadings.shape[1])  # Avoid out-of-range slice
        importance_scores = loadings[:, :k].sum(axis=1)

        # Step 4: Select top 10 columns
        num_to_select = min(10, len(importance_scores))  # Safe guard
        top_indices = np.argsort(importance_scores)[-num_to_select:][::-1]
        top_columns = data_scaled.columns[top_indices].tolist()

        if verbose:
            print("Top columns selected:", top_columns)
            print(f"Total variance explained with the 10 columns: {pca.explained_variance_ratio_[:num_to_select].sum():.2%}")
        return df[top_columns]


    def save_selected_areas_to_csv(self, filepath: str):
        """
        Save the selected feature (area) columns after dimensionality reduction to a timestamped CSV file.

        This method:
        - Ensures the directory exists.
        - Appends a timestamp to the filename to avoid overwriting.
        - Saves the names of the columns in `self.combined` (i.e., the areas kept) to a CSV file.

        Args:
            filepath (str): The base filepath (including directory and filename) where the CSV should be saved.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Add timestamp before file extension
        base, ext = os.path.splitext(filepath)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath_with_timestamp = f"{base}_{self.reduction_type}_{timestamp}{ext}"

        # Extract selected column names
        selected_columns = self.combined.columns.tolist()
        df = pd.DataFrame({'selected_areas': selected_columns})
        df.to_csv(filepath_with_timestamp, index=False)

        print(f"Saved selected area columns to {filepath_with_timestamp}")

