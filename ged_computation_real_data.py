import os
import json
import networkx as nx
import numpy as np
import pandas as pd
from utils import *  # type: ignore
from ot.gromov import (
    fused_gromov_wasserstein,
    entropic_fused_gromov_wasserstein,
    BAPG_fused_gromov_wasserstein,
)

from time import time
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd




# def load_and_process_dataframe(df, dataset_name):
#     print(f"Loading dataframe for dataset: {dataset_name}")
#     if 'graph_edge_list' not in df.columns:
#         raise ValueError("The dataframe does not contain the 'graph_edge_list' column.")
    
#     for index, row in df.iterrows():
#         edge_list = json.loads(row['graph_edge_list'])  # Assuming the edge list is stored as a JSON string
#         graph = nx.Graph()
#         graph.add_edges_from(edge_list)
        

#     if dataset_name == 'AIDS':
#         for index, row in df.iterrows():
#             if 'node_labels' in row:
#                 labels = df['node_labels']  # Assuming labels are stored as a JSON string
#             else:
#                 print(f"Row {index} does not contain labels.")
#     #return graph




def compare_and_swap_graphs(G_S, G_T, labels_S = None, labels_T = None):
    #print(f"G_S nodes: {G_S.number_of_nodes()}, G_T nodes: {G_T.number_of_nodes()}")
    if G_S.number_of_nodes() > G_T.number_of_nodes():
        G_S, G_T = G_T, G_S
        #print("Graphs swapped.")
        #print(f"G_S nodes: {G_S.number_of_nodes()}, G_T nodes: {G_T.number_of_nodes()}")

    # Add dummy nodes to the smaller graph
    existing_nodes = set(G_S.nodes())
    next_dummy_id = max(existing_nodes, default=-1) + 1 if existing_nodes else 0

    while G_S.number_of_nodes() < G_T.number_of_nodes():
        dummy_node = next_dummy_id
        G_S.add_node(dummy_node)
        next_dummy_id += 1
    
    if labels_S is not None and labels_T is not None:
        while len(labels_S) < len(labels_T):
            labels_S.append("?")
    if labels_T is not None and labels_S is not None:
        return G_S, G_T, labels_S, labels_T
    else:
        return G_S, G_T
    
def compute_label_distance(labels_S, labels_T, number_of_nodes):
    if labels_S is None and labels_T is None:
        return np.ones((number_of_nodes, number_of_nodes))
    
    if len(labels_S) != len(labels_T):
        raise ValueError("labels_S and labels_T must have the same length.")
        
    label_distance_matrix = np.zeros((len(labels_S), len(labels_T)))
    for i, label_S in enumerate(labels_S):
        for j, label_T in enumerate(labels_T):
            label_distance_matrix[i][j] = 0.0 if label_S == label_T else 2.0
        
    return label_distance_matrix

# def compute_cross_cost_matrix_no_structural_features(G2, G3, N2, N3):
#     F2 = np.zeros((N2, 1))
#     for i in range(N2):
#         F2[i, 0] = 1 if G2.nodes[i]['label'] == 1 else 0
#     F3 = np.zeros((N3, 1))
#     for i in range(N3):
#         F3[i, 0] = 1 if G3.nodes[i]['label'] == 1 else 0
    

#     # Compute pairwise euclidean distance between node features
#     M = (F2**2).dot(np.ones((1, N3))) + np.ones((N2, 1)).dot((F3**2).T) - 2 * F2.dot(F3.T)
#     return F2, F3, M

# def compute_structural_features(graph):
#     degree_centrality = np.array(list(networkx.degree_centrality(graph).values()))
#     pagerank_centrality = np.array(list(networkx.pagerank(graph).values()))
#     clustering_coefficient = np.array(list(networkx.clustering(graph).values()))
#     return np.vstack((degree_centrality, pagerank_centrality, clustering_coefficient)).T

# def compute_cross_matrix_with_structural_features(G2, G3, N2, N3):

#     F2 = np.zeros((N2, 1))
#     for i in range(N2):
#         F2[i, 0] = 1 if G2.nodes[i]['label'] == 1 else 0
    
#     S2 = compute_structural_features(G2)

#     F3 = np.zeros((N3, 1))
#     for i in range(N3):
#         F3[i, 0] = 1 if G3.nodes[i]['label'] == 1 else 0
    

#     S3 = compute_structural_features(G3)

#     # Concatenate F2 and S2
#     F2 = np.hstack((F2, S2))

#     # Concatenate F3 and S3
#     F3 = np.hstack((F3, S3))

#     # Compute pairwise euclidean distance between node features
#     M = np.sum(F2**2, axis=1, keepdims=True).dot(np.ones((1, F3.shape[0]))) + \
#         np.ones((F2.shape[0], 1)).dot(np.sum(F3**2, axis=1, keepdims=True).T) - 2 * F2.dot(F3.T)
    
#     return F2, F3, M
    
def compute_structural_features(graph):
    degree_centrality = np.array(list(nx.degree_centrality(graph).values()))
    pagerank_centrality = np.array(list(nx.pagerank(graph).values()))
    clustering_coefficient = np.array(list(nx.clustering(graph).values()))
    return np.vstack((degree_centrality, pagerank_centrality, clustering_coefficient)).T


def compute_cross_matrix_with_structural_features(G1, G2):
    N1 = G1.number_of_nodes()
    N2 = G2.number_of_nodes()
    S1 = compute_structural_features(G1)
    S2 = compute_structural_features(G2)
    
    if S1.shape[1] != S2.shape[1]:
        raise ValueError(f"Structural features of G1 and G2 must have the same number of columns. Got {S1.shape[1]} and {S2.shape[1]}.")
    # Compute pairwise Euclidean distance between rows of S1 and S2
    structural_cross_matrix = np.sum(S1**2, axis=1, keepdims=True).dot(np.ones((1, N2))) + \
                      np.ones((N1, 1)).dot(np.sum(S2**2, axis=1, keepdims=True).T) - \
                      2 * S1.dot(S2.T)
    
    
    return structural_cross_matrix


    
def calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu, include_structural_features):
    if not include_structural_features:
        return label_distance_matrix
    return label_distance_matrix + mu * structural_cross_matrix


def linear_regression(X_train, y_train, X_test, y_test, scale_features=False):
    """
    Performs linear regression using sklearn's LinearRegression and computes MAE and custom accuracy.

    Args:
        X_train (numpy array): Training data features.
        y_train (numpy array): Training data target.
        X_test (numpy array): Testing data features.
        y_test (numpy array): Testing data target.
        scale_features (bool): Whether to scale the features using StandardScaler.

    Returns:
        tuple: A tuple containing:
            - mae (float): Mean Absolute Error on the test data.
            - accuracy (float): Custom accuracy on the test data.
    """

    # Feature Scaling
    if scale_features:
        print("Scaling features...")
        scaler_X = StandardScaler()
        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
        y_test = scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).flatten()
    else:
        print("No scaling applied.")


    # Create a Linear Regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict the values for the test data
    y_pred = model.predict(X_test)

     # Inverse transform the prediction
    if scale_features:
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate the Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate the custom accuracy
    rounded_predictions = np.round(y_pred)
    correct_predictions = np.abs(y_test - rounded_predictions) <= 1
    accuracy = np.mean(correct_predictions) * 100  # as percentage
    print(f"Linear Regression Mean Absolute Error: {mae}")
    print(f"Accuracy: {accuracy:.2f}%")
    # Compute Spearman's correlation coefficient
    spearman_corr = pd.Series(y_test).corr(pd.Series(y_pred), method='spearman')
    print(f"Spearman's Correlation Coefficient: {spearman_corr:.4f}")

    # Compute Kendall's tau coefficient
    kendall_tau = pd.Series(y_test).corr(pd.Series(y_pred), method='kendall')
    print(f"Kendall's Tau Coefficient: {kendall_tau:.4f}")
    print(10*'*')

def support_vector_regression(X_train, y_train, X_test, y_test):
    # Feature scaling is important for SVR
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).flatten()

    # Support Vector Regression
    for kernel_type in ['rbf', 'linear', 'poly']:
        #kernel = 'rbf'  # You can try 'linear', 'poly', 'rbf'
        print(f"Using kernel: {kernel_type}")
        model = SVR(kernel=kernel_type, C=1.0, epsilon=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        #mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        #print(f"SVR Mean Squared Error: {mse}")
        print(f"SVR Mean Absolute Error: {mae}")
        accuracy = np.mean(np.abs(np.round(y_pred) - y_test) <= 1) * 100
        print(f"Accuracy: {accuracy:.2f}%")
        # Compute Spearman's correlation coefficient
        spearman_corr = pd.Series(y_test).corr(pd.Series(y_pred), method='spearman')
        print(f"Spearman's Correlation Coefficient: {spearman_corr:.4f}")

        # Compute Kendall's tau coefficient
        kendall_tau = pd.Series(y_test).corr(pd.Series(y_pred), method='kendall')
        print(f"Kendall's Tau Coefficient: {kendall_tau:.4f}")
        print(10*'*')



def compute_ged_GW(G1, G2, cross_matrix):
    if G1.number_of_nodes() != G2.number_of_nodes():
        raise ValueError(f"G1 and G2 must have the same number of nodes. Got {G1.number_of_nodes()} and {G2.number_of_nodes()}.")
    normalization_factor = max(G1.number_of_nodes(), G2.number_of_nodes()) + max(G1.number_of_edges(), G2.number_of_edges())
    C1 = nx.to_numpy_array(G1)
    C2 = nx.to_numpy_array(G2)
   
    h1 = np.ones(G1.number_of_nodes()) / G1.number_of_nodes()
    h2 = np.ones(G2.number_of_nodes()) / G2.number_of_nodes()

    _, log_cg = fused_gromov_wasserstein(
                cross_matrix, C1, C2, h1, h2, "square_loss", alpha=0.5, tol_rel=1e-9, verbose=False, log=True
            )
    
    return log_cg['fgw_dist'], log_cg['fgw_dist']*normalization_factor
    
def GED_prediction_regression(file_name):
    # Open the DataFrame from the given file name
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return
    
    # Split the data into training and test sets (80% training, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


    X_train = train_df[['GW_Score']]
    y_train = train_df['True_GED']
    X_test = test_df[['GW_Score']]
    y_test = test_df['True_GED']

    linear_regression(X_train, y_train, X_test, y_test, scale_features=False)
    linear_regression(X_train, y_train, X_test, y_test, scale_features=True)
    support_vector_regression(X_train, y_train, X_test, y_test)


def partial_timing(G_1, G_2, dataset, structural_features, id_1, id_2):
        if dataset == 'AIDS':
            labels_dict = label_dict_construction()
            labels_1 = labels_dict.get(id_1, None)
            labels_2 = labels_dict.get(id_2, None)
            
            
            if labels_1 is not None and len(labels_1) != G_1.number_of_nodes():
                raise ValueError(f"Length of L1 ({len(L1)}) does not match the number of nodes in G_1 ({G_1.number_of_nodes()}).")
            if labels_2 is not None and len(labels_2) != G_2.number_of_nodes():
                raise ValueError(f"Length of L2 ({len(L2)}) does not match the number of nodes in G_2 ({G_2.number_of_nodes()}).")
            G_1, G_2, labels_1, labels_2 = compare_and_swap_graphs(G_1, G_2, labels_1, labels_2)
            #label_distance_matrix = compute_label_distance(labels_1, labels_2, G_1.number_of_nodes())
            #structural_cross_matrix = compute_cross_matrix_with_structural_features(G_1, G_2)
            h1 = np.ones(G_1.number_of_nodes()) / G_1.number_of_nodes()
            h2 = np.ones(G_2.number_of_nodes()) / G_2.number_of_nodes()
            C1 = nx.to_numpy_array(G_1)
            C2 = nx.to_numpy_array(G_2)
            start_time = time()
            if structural_features:
                label_distance_matrix = compute_label_distance(labels_1, labels_2, G_1.number_of_nodes())
                structural_cross_matrix = compute_cross_matrix_with_structural_features(G_1, G_2)
                cross_matrix = label_distance_matrix + 0.5 * structural_cross_matrix
            
            else:
                cross_matrix = compute_label_distance(labels_1, labels_2, G_1.number_of_nodes())
            T_cg, log_cg = fused_gromov_wasserstein(
                cross_matrix, C1, C2, h1, h2, "square_loss", alpha=0.5, tol_rel=1e-6, verbose=False, log=True
                )
            end_time = time()
            return end_time-start_time
        else:
            G_1, G_2 = compare_and_swap_graphs(G_1, G_2)
            h1 = np.ones(G_1.number_of_nodes()) / G_1.number_of_nodes()
            h2 = np.ones(G_2.number_of_nodes()) / G_2.number_of_nodes()
            C1 = nx.to_numpy_array(G_1)
            C2 = nx.to_numpy_array(G_2)
            #label_distance_matrix = compute_label_distance(None, None, G_1.number_of_nodes())
            #structural_cross_matrix = compute_cross_matrix_with_structural_features(G_1, G_2)
            start_time = time()
            
            if structural_features:
                label_distance_matrix = compute_label_distance(None, None, G_1.number_of_nodes())
                structural_cross_matrix = compute_cross_matrix_with_structural_features(G_1, G_2)
                cross_matrix = label_distance_matrix + 0.5 * structural_cross_matrix
            else:
                cross_matrix = compute_label_distance(None, None, G_1.number_of_nodes())
          
            T_cg, log_cg = fused_gromov_wasserstein(
                cross_matrix, C1, C2, h1, h2, "square_loss", alpha=0.5, tol_rel=1e-6, verbose=False, log=True
                )
            end_time = time()
            return end_time-start_time
            
            
def partial_timing_k_graphs(k, dataset, dataset_path_true_ged, dataset_path_graphs, structural_features, number_iterations):
        if os.path.exists(dataset_path_true_ged):
            true_ged_df = pd.read_csv(dataset_path_true_ged)
        else:
            print(f"Dataset file not found: {dataset_path_true_ged}")
        dataset_path_graphs = os.path.join("Dataset", dataset, f"{dataset}_graphs.csv")
        if os.path.exists(dataset_path_graphs):
            graphs_df = pd.read_csv(dataset_path_graphs)
        else:
            print(f"Dataset file not found: {dataset_path_graphs}")
        cumulative_times = []
        for _ in range(number_iterations):  # Repeat 10 times
            sampled_true_ged_df = true_ged_df.sample(n=k)
            cumulative_time = 0.0
            for index, row in sampled_true_ged_df.iterrows():
                id_1, id_2, _ = row['id_1'], row['id_2'], row['true_ged']
                graph_1 = graphs_df[graphs_df['graph_id'] == id_1]
                edge_list_1 = json.loads(graph_1.iloc[0]['graph_edge_list'])
                G_1 = nx.Graph()
                G_1.add_edges_from(edge_list_1)
                graph_2 = graphs_df[graphs_df['graph_id'] == id_2]
                edge_list_2 = json.loads(graph_2.iloc[0]['graph_edge_list'])
                G_2 = nx.Graph()
                G_2.add_edges_from(edge_list_2)
                cumulative_time += partial_timing(G_1, G_2, dataset, structural_features, id_1, id_2)
            cumulative_times.append(cumulative_time)
        return np.mean(cumulative_times)
        

    
    # Define the range of k values
def evaluate_timing_for_datasets():
    """
    Evaluates the timing for different datasets and structural feature configurations
    across a range of k values. Saves the results to CSV files.
    """
    k_values = np.linspace(50, 500, 10, dtype=int)
    results = []

    for dataset in ['AIDS', 'IMDB', 'Linux']:
        for structural_features in [False, True]:
            dataset_path_true_ged = os.path.join("True_GED", dataset, f"{dataset}_ged.csv")
            dataset_path_graphs = os.path.join("Dataset", dataset, f"{dataset}_graphs.csv")
            
            timing_results = []

            for k in k_values:
                # Compute timing for each k
                avg_time = partial_timing_k_graphs(k, dataset, dataset_path_true_ged, dataset_path_graphs, structural_features, number_iterations=5)
                timing_results.append({'k': k, 'Avg_Time': avg_time})
            
            # Convert results to DataFrame
        results_df = pd.DataFrame(timing_results)
        print(results_df)
            
            # Save to CSV
        feature_suffix = 'with_features' if structural_features else 'no_features'
        output_path = f'risultati/timing_{dataset}_{feature_suffix}.csv'
        results_df.to_csv(output_path, index=False)


def make_plots():        
    """
    Generates and saves plots for timing results from CSV files in the 'risultati' folder.
    The plots are saved in the 'figure' folder.
    """
    
    datasets = ['AIDS', 'IMDB', 'Linux']
    feature_suffixes = ['with_features', 'no_features']
    input_folder = 'risultati'
    output_folder = 'figure'
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    for dataset in datasets:
        # Combine the two feature suffixes into a single plot
        df_with_features = None
        df_no_features = None

        # Load dataframes for both feature suffixes
        for suffix in feature_suffixes:
            file_name = f'timing_{dataset}_{suffix}.csv'
            file_path = os.path.join(input_folder, file_name)
            if os.path.exists(file_path):
                if suffix == 'with_features':
                    df_with_features = pd.read_csv(file_path)
                elif suffix == 'no_features':
                    df_no_features = pd.read_csv(file_path)

        # If both dataframes are available, plot them together
        if df_with_features is not None and df_no_features is not None:
            plt.figure()
            plt.plot(df_with_features['k'], df_with_features['Avg_Time'], marker='o', markersize=8, linestyle='--', linewidth=2, label='With Features')
            plt.plot(df_no_features['k'], df_no_features['Avg_Time'], marker='s', markersize=8, linestyle='-', linewidth=2, label='No Features')
            plt.xlabel('Number of Pairs of Graphs', fontsize = 13)
            plt.ylabel('Average Time (seconds)', fontsize = 13)
            #plt.title(f'Timing Results for {dataset}')
            plt.legend()
            plt.grid(True)

            # Save the combined plot
            output_file = os.path.join(output_folder, f'timing_{dataset}_combined.png')
            plt.savefig(output_file)
            plt.close()
            print(f"Combined plot saved: {output_file}")
        else:
            print(f"Dataframes for combined plot not found for dataset: {dataset}")
        
evaluate_timing_for_datasets()
make_plots()

    
    



dataset_names = ['AIDS', 'IMDB', 'Linux']

for dataset in dataset_names:
    print(dataset)
    
    results_no_structural_features = pd.DataFrame({'GW_Score': pd.Series(dtype='float'), 
                                                   'Normalized_GW_Score': pd.Series(dtype='float'), 
                                                   'True_GED': pd.Series(dtype='float'),
                                                   'Normalized_True_GED': pd.Series(dtype='float')})
    results_with_structural_features = pd.DataFrame({'GW_Score': pd.Series(dtype='float'), 
                                                     'Normalized_GW_Score': pd.Series(dtype='float'), 
                                                     'True_GED': pd.Series(dtype='float'),
                                                     'Normalized_True_GED': pd.Series(dtype='float')})
    dataset_path_true_ged = os.path.join("True_GED", dataset, f"{dataset}_ged.csv")
    if os.path.exists(dataset_path_true_ged):
        true_ged_df = pd.read_csv(dataset_path_true_ged)
        sampled_true_ged_df = true_ged_df.sample(n=1000)
    else:
        print(f"Dataset file not found: {dataset_path_true_ged}")
    dataset_path_graphs = os.path.join("Dataset", dataset, f"{dataset}_graphs.csv")
    if os.path.exists(dataset_path_graphs):
        graphs_df = pd.read_csv(dataset_path_graphs)
    else:
        print(f"Dataset file not found: {dataset_path_graphs}")
    #for index, row in true_ged_df.iterrows():
    for index, row in sampled_true_ged_df.iterrows():
        id_1, id_2, true_ged = row['id_1'], row['id_2'], row['true_ged']
        graph_1 = graphs_df[graphs_df['graph_id'] == id_1]
        edge_list_1 = json.loads(graph_1.iloc[0]['graph_edge_list'])
        G_1 = nx.Graph()
        G_1.add_edges_from(edge_list_1)
        graph_2 = graphs_df[graphs_df['graph_id'] == id_2]
        edge_list_2 = json.loads(graph_2.iloc[0]['graph_edge_list'])
        G_2 = nx.Graph()
        G_2.add_edges_from(edge_list_2)
        normalization_factor = max(G_1.number_of_nodes(), G_2.number_of_nodes()) + max(G_1.number_of_edges(), G_2.number_of_edges())
        normalized_ged = float(true_ged)/normalization_factor
        if dataset == 'AIDS':
            labels_dict = label_dict_construction()
            labels_1 = labels_dict.get(id_1, None)
            labels_2 = labels_dict.get(id_2, None)
           
            if labels_1 is not None and len(labels_1) != G_1.number_of_nodes():
                raise ValueError(f"Length of L1 ({len(L1)}) does not match the number of nodes in G_1 ({G_1.number_of_nodes()}).")
            if labels_2 is not None and len(labels_2) != G_2.number_of_nodes():
                raise ValueError(f"Length of L2 ({len(L2)}) does not match the number of nodes in G_2 ({G_2.number_of_nodes()}).")
            G_1, G_2, labels_1, labels_2 = compare_and_swap_graphs(G_1, G_2, labels_1, labels_2)
            label_distance_matrix = compute_label_distance(labels_1, labels_2, G_1.number_of_nodes())
            structural_cross_matrix = compute_cross_matrix_with_structural_features(G_1, G_2)
            
            cross_matrix_no_structural_features = calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu=0.5, include_structural_features=False)
            gw_score, normalized_gw_score = compute_ged_GW(G_1, G_2, cross_matrix_no_structural_features)
            results_no_structural_features = pd.concat([results_no_structural_features, pd.DataFrame({'GW_Score': [gw_score], 'Normalized_GW_Score': [normalized_gw_score], 'True_GED': [true_ged], 'Normalized_True_GED': normalized_ged})], ignore_index=True)
            
            cross_matrix_with_structural_features = calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu=0.5, include_structural_features=True)
            gw_score, normalized_gw_score = compute_ged_GW(G_1, G_2, cross_matrix_with_structural_features)
            results_with_structural_features = pd.concat([results_with_structural_features, pd.DataFrame({'GW_Score': [gw_score], 'Normalized_GW_Score': [normalized_gw_score], 'True_GED': [true_ged], 'Normalized_True_GED': normalized_ged})], ignore_index=True)
            
        else:
            G_1, G_2 = compare_and_swap_graphs(G_1, G_2)
            label_distance_matrix = compute_label_distance(None, None, G_1.number_of_nodes())
            structural_cross_matrix = compute_cross_matrix_with_structural_features(G_1, G_2)

            cross_matrix_no_structural_features = calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu=0.5, include_structural_features=False)
            gw_score, normalized_gw_score = compute_ged_GW(G_1, G_2, cross_matrix_no_structural_features)
            results_no_structural_features = pd.concat([results_no_structural_features, pd.DataFrame({'GW_Score': [gw_score], 'Normalized_GW_Score': [normalized_gw_score], 'True_GED': [true_ged], 'Normalized_True_GED': normalized_ged})], ignore_index=True)
            
            cross_matrix_with_structural_features = calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu=0.5, include_structural_features=True)
            gw_score, normalized_gw_score = compute_ged_GW(G_1, G_2, cross_matrix_with_structural_features)
            results_with_structural_features = pd.concat([results_with_structural_features, pd.DataFrame({'GW_Score': [gw_score], 'Normalized_GW_Score': [normalized_gw_score], 'True_GED': [true_ged], 'Normalized_True_GED': normalized_ged})], ignore_index=True)
    
    #print("Results without structural features:")
    #print(results_no_structural_features)
    #results_no_structural_features.to_csv(f'risultati/{dataset}_no_structural_features.csv', index=False)
    #print("Results with structural features:")
    #print(results_with_structural_features)
    #results_with_structural_features.to_csv(f'risultati/{dataset}_with_structural_features.csv', index=False)

# Iterate over datasets and process results
for dataset in dataset_names:
    print(f"Processing dataset: {dataset}")
    # Process results without structural features
    no_structural_features_path = f'risultati/{dataset}_no_structural_features.csv'
    if os.path.exists(no_structural_features_path):
        print(f"Processing {no_structural_features_path}")
        GED_prediction_regression(no_structural_features_path)
    else:
        print(f"File not found: {no_structural_features_path}")

        # Process results with structural features
    with_structural_features_path = f'risultati/{dataset}_with_structural_features.csv'
    if os.path.exists(with_structural_features_path):
        print(f"Processing {with_structural_features_path}")
        GED_prediction_regression(with_structural_features_path)
    else:
        print(f"File not found: {with_structural_features_path}")
