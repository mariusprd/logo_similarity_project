import pandas as pd
import os
from sklearn.cluster import KMeans, DBSCAN
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ast import literal_eval

class Grouper:
    def __init__(self):
        """
        Initialize the Grouper with data to be grouped.
        """
        self.features = None
        self.results = None
        self.similarity_matrix_dir = None

    def group_by(self, criteria, algorithm, num_clusters=None, features_filepath=None, results_filepath=None):
        """
        General method to group data based on the provided criteria and algorithm.
        
        :param criteria: A string indicating the grouping criteria ('overall', 'top2', 'all_colors', 'phash', or 'conv_features').
        :param algorithm: A string specifying the grouping algorithm ('dbscan', 'kmeans', or 'graph_based').
        :param num_clusters: (Optional; required for kmeans) An integer specifying the number of clusters.
        :param features_filepath: Optional path to a CSV file containing features.
        :param results_filepath: Path to the CSV file where results will be written.
        :return: The grouping result.
        """
        # Validate the criteria
        if criteria not in ['overall', 'top2', 'all_colors', 'phash', 'conv_features']:
            raise ValueError("Invalid criteria. Must be one of 'overall', 'top2', 'all_colors', 'phash', or 'conv_features'.")

        # Validate the algorithm
        if algorithm not in ['dbscan', 'kmeans', 'graph_based']:
            raise ValueError("Invalid algorithm. Must be one of 'dbscan', 'kmeans', or 'graph_based'.")

        # For kmeans, ensure num_clusters is provided and valid
        if algorithm == 'kmeans':
            if num_clusters is None:
                raise ValueError("For kmeans algorithm, 'num_clusters' parameter is required.")
            try:
                num_clusters = int(num_clusters)
            except ValueError:
                raise ValueError("'num_clusters' must be an integer.")
            
        if features_filepath is not None:
            self.features = pd.read_csv(features_filepath)
        else:
            if criteria == 'phash' and os.path.exists('./data/phash.csv'):
                self.features = pd.read_csv('./data/phash.csv')
            elif criteria == 'conv_features' and os.path.exists('./data/convnet_features.csv'):
                self.features = pd.read_csv('./data/convnet_features.csv')
            elif criteria == 'overall' and os.path.exists('./data/color_palette_1color.csv'):
                self.features = pd.read_csv('./data/color_palette_1color.csv')
            elif (criteria == 'top2' or criteria == 'all_colors') and os.path.exists('./data/color_palette_4color.csv'):
                self.features = pd.read_csv('./data/color_palette_4color.csv')
            else:
                raise ValueError("No features file provided or available for the specified criteria.")
                
        self.results = results_filepath

        # Call the corresponding grouping method
        if algorithm == 'kmeans':
            return self._group_by_kmeans(criteria, num_clusters)
        elif algorithm == 'dbscan':
            return self._group_by_dbscan(criteria)
        elif algorithm == 'graph_based':
            return self._group_by_graph_based(criteria)
        else:
            raise NotImplementedError("The provided algorithm is not implemented.")

    def _parse_color(self, color_str):
        """
        Convert a string like "(35, 78, 122)" into a NumPy array.
        """
        return np.array(literal_eval(color_str))

    def _group_by_kmeans(self, criteria, num_clusters):
        """
        Group data using k-means clustering based on the specified criteria, and
        write the results to the file specified by self.results in the format:
        Filepath, <criteria>
        """
        # Determine features based on the criteria
        if criteria in ['phash', 'conv_features']:
            # Use all features except the 'Filepath' column
            features = self.features.drop(columns=['Filepath'])
        elif criteria in ['all_colors']:
            # Concatenate all color columns (parsing each string) into one numeric vector per row.
            color_cols = self.features.filter(regex=r'^Color_\d+$')
            features = color_cols.apply(lambda row: np.concatenate([self._parse_color(x) for x in row]), axis=1)
            features = np.vstack(features.tolist())
        elif criteria == 'top2':
            # Identify confidence columns (assumed to be in the format 'Confidence_<number>')
            conf_cols = self.features.filter(regex=r'^Confidence_\d+$').columns

            def get_top2(row):
                # Get the top 2 confidence column names
                top_conf_cols = row[conf_cols].nlargest(2).index
                # Map them to the corresponding color column names and parse
                colors = [self._parse_color(row[col.replace('Confidence_', 'Color_')]) for col in top_conf_cols]
                # Concatenate the two color arrays into one vector
                return np.concatenate(colors)
            features = self.features.apply(get_top2, axis=1)
            features = np.vstack(features.tolist())
        elif criteria == 'overall':
            # Use only the 'Color_1' column (parsed as a numeric array)
            color_values = self.features['Color_1'].apply(self._parse_color).tolist()
            features = np.vstack(color_values)
        else:
            raise ValueError("Unsupported criteria for kmeans clustering.")

        # Perform KMeans clustering on the selected features
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(features)

        # Prepare results: create a DataFrame with the Filepath column and a new column with the criteria name containing the group labels
        if 'Filepath' in self.features.columns:
            output_df = pd.DataFrame({
                'Filepath': self.features['Filepath'],
                criteria: labels
            })
        else:
            output_df = pd.DataFrame({
                'Filepath': self.features.index,
                criteria: labels
            })

        output_df.to_csv(self.results, index=False)
        return output_df

    def _group_by_dbscan(self, criteria):
        """
        Group data using DBSCAN clustering based on the specified criteria, and
        write the results to the file specified by self.results in the format:
        Filepath, <criteria>
        """
        # Determine features based on the criteria
        if criteria in ['phash', 'conv_features']:
            features = self.features.drop(columns=['Filepath'])
        elif criteria in ['all_colors']:
            color_cols = self.features.filter(regex=r'^Color_\d+$')
            features = color_cols.apply(lambda row: np.concatenate([self._parse_color(x) for x in row]), axis=1)
            features = np.vstack(features.tolist())
        elif criteria == 'top2':
            conf_cols = self.features.filter(regex=r'^Confidence_\d+$').columns

            def get_top2(row):
                top_conf_cols = row[conf_cols].nlargest(2).index
                colors = [self._parse_color(row[col.replace('Confidence_', 'Color_')]) for col in top_conf_cols]
                return np.concatenate(colors)
            features = self.features.apply(get_top2, axis=1)
            features = np.vstack(features.tolist())
        elif criteria == 'overall':
            color_values = self.features['Color_1'].apply(self._parse_color).tolist()
            features = np.vstack(color_values)
        else:
            raise ValueError("Unsupported criteria for DBSCAN clustering.")

        # Perform DBSCAN clustering on the selected features.
        dbscan = DBSCAN(eps=12, min_samples=1)
        labels = dbscan.fit_predict(features)

        if 'Filepath' in self.features.columns:
            output_df = pd.DataFrame({
                'Filepath': self.features['Filepath'],
                criteria: labels
            })
        else:
            output_df = pd.DataFrame({
                'Filepath': self.features.index,
                criteria: labels
            })

        output_df.to_csv(self.results, index=False)
        return output_df

    def _group_by_graph_based(self, criteria, similarity_threshold=None):
        """
        Group data using a graph-based approach and write the results to the file specified by self.results
        in the format: Filepath, <criteria>.
        
        This method recomputes the entire similarity matrix every time it is called.
        """
        # Determine similarity function and default threshold based on criteria.
        if criteria == 'conv_features':
            if similarity_threshold is None:
                similarity_threshold = 0.89
            features = (self.features.drop(columns=['Filepath']).to_numpy() 
                        if 'Filepath' in self.features.columns else self.features.to_numpy())
            def cosine_similarity(a, b):
                return np.dot(a, b)
            sim_func = cosine_similarity
        elif criteria == 'phash':
            if similarity_threshold is None:
                similarity_threshold = 15
            if 'pHash' not in self.features.columns:
                raise ValueError("Data must contain a 'pHash' column for phash-based grouping.")
            # Convert hex string pHash values to Python integers.
            features = self.features['pHash'].apply(lambda x: int(x, 16)).tolist()
            def phash_diff(a, b):
                return abs(a - b)
            sim_func = phash_diff
        else:
            raise ValueError("Unsupported criteria for graph-based grouping.")

        # Build list of file identifiers.
        file_ids = self.features['Filepath'].tolist() if 'Filepath' in self.features.columns else list(self.features.index)
        total_files = len(file_ids)
        
        # Recompute the entire similarity matrix.
        sim_matrix = self._update_similarity_matrix(features, sim_func, criteria)
        
        # Create a mapping from file id to index.
        id_to_idx = {fid: idx for idx, fid in enumerate(file_ids)}

        # Perform greedy grouping.
        groups = []
        for i in range(total_files):
            fid = file_ids[i]
            added_to_group = False
            for group in groups:
                if criteria == 'conv_features':
                    condition = all(sim_matrix[i, id_to_idx[member]] >= similarity_threshold for member in group)
                elif criteria == 'phash':
                    condition = all(sim_matrix[i, id_to_idx[member]] < similarity_threshold for member in group)
                else:
                    condition = False

                if condition:
                    group.append(fid)
                    added_to_group = True
                    break
            if not added_to_group:
                groups.append([fid])

        # Assign group labels.
        file_to_group = {}
        for group_idx, group in enumerate(groups, start=1):
            for f in group:
                file_to_group[f] = group_idx

        output_df = pd.DataFrame({
            'Filepath': file_ids,
            criteria: [file_to_group[f] for f in file_ids]
        })
        output_df.to_csv(self.results, index=False)
        return output_df


    def _update_similarity_matrix(self, features, similarity_func, criteria):
        """
        Compute a symmetric similarity matrix for the given features using the provided similarity function.
        
        This function recomputes the entire similarity matrix without reading or writing to any file.
        """
        # Determine the number of items.
        if hasattr(features, "shape"):
            n = features.shape[0]
        else:
            n = len(features)

        sim_matrix = np.zeros((n, n))

        def compute_pair(i, j):
            return similarity_func(features[i], features[j])

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            futures = {}
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        continue
                    futures[(i, j)] = executor.submit(compute_pair, i, j)
            for (i, j), future in futures.items():
                sim = future.result()
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        # Set diagonal elements.
        for i in range(n):
            sim_matrix[i, i] = 1 if criteria == 'conv_features' else 0
        return sim_matrix