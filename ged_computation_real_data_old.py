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
            label_distance_matrix[i][j] = 0 if label_S == label_T else 1
        
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
    print(S1)
    print(S2)
    
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
    
    '''
    normalized_app_ged = []
    gromow_wassertstein_scores = []
    normalization_factor = []

    for _ in range(M):
        G2, G3, app_ged = generate_graphs(n, p)
        ged_values.append(app_ged)
        normalized_ged = app_ged / (max(G2.number_of_nodes(), G3.number_of_nodes()) + max(G2.number_of_edges(), G3.number_of_edges()))
        normalized_app_ged.append(normalized_ged)
        normalization_factor.append(
            max(G2.number_of_nodes(), G3.number_of_nodes()) + max(G2.number_of_edges(), G3.number_of_edges())
        )

        C2 = networkx.to_numpy_array(G2)
        C3 = networkx.to_numpy_array(G3)

        h2 = np.ones(n) / n
        h3 = np.ones(n) / n

        if structural_features:
            F1, F2, M = compute_cross_matrix_with_structural_features(G2, G3, n, n)
        else:
            F1, F2, M = compute_cross_cost_matrix_no_structural_features(G2, G3, n, n)

        _, log_cg = fused_gromov_wasserstein(
                M, C2, C3, h2, h3, "square_loss", alpha=0.5, tol_rel=1e-9, verbose=False, log=True
            )
        gromow_wassertstein_scores.append(log_cg)

    # Create a DataFrame with the required columns
    df = pd.DataFrame({
        'GED': ged_values,
        'Normalized GED': normalized_app_ged,
        'Gromov-Wasserstein Score': [log['fgw_dist'] for log in gromow_wassertstein_scores],
        'Normalization Factor': normalization_factor
    })
    df['Gromow_Wasserstein_Score_Normalized'] = df['Gromov-Wasserstein Score'] * df['Normalization Factor']
    #print(df)
    # Print the DataFrame
    # Save the DataFrame to a CSV file in the "risultati" folder
    if structural_features:
        df.to_csv('risultati/regression-ged_results_structural_features.csv', index=False)
    else:
        df.to_csv('risultati/regression-ged_results_no_structural_features.csv', index=False)
    '''




dataset_names = ['AIDS', 'IMDB', 'Linux']
#dataset_names = ['AIDS'] 
#dataset_names = ['IMDB', 'Linux']
for dataset in dataset_names:
    print(dataset)
    dataset_path_true_ged = os.path.join("True_GED", dataset, f"{dataset}_ged.csv")
    if os.path.exists(dataset_path_true_ged):
        true_ged_df = pd.read_csv(dataset_path_true_ged)
        sampled_true_ged_df = true_ged_df.sample(n=1)
    else:
        print(f"Dataset file not found: {dataset_path_true_ged}")
    dataset_path_graphs = os.path.join("Dataset", dataset, f"{dataset}_graphs.csv")
    if os.path.exists(dataset_path_graphs):
        graphs_df = pd.read_csv(dataset_path_graphs)
    else:
        print(f"Dataset file not found: {dataset_path_graphs}")
   
    for index, row in sampled_true_ged_df.iterrows():
        id_1, id_2 = row['id_1'], row['id_2']
        graph_1 = graphs_df[graphs_df['graph_id'] == id_1]
        edge_list_1 = json.loads(graph_1.iloc[0]['graph_edge_list'])
        G_1 = nx.Graph()
        G_1.add_edges_from(edge_list_1)
        graph_2 = graphs_df[graphs_df['graph_id'] == id_2]
        edge_list_2 = json.loads(graph_2.iloc[0]['graph_edge_list'])
        G_2 = nx.Graph()
        G_2.add_edges_from(edge_list_2)
        print(f"G_1: {G_1.number_of_nodes()} nodes, {G_1.number_of_edges()} edges")
        print(f"G_2: {G_2.number_of_nodes()} nodes, {G_2.number_of_edges()} edges")
        if dataset == 'AIDS':
            labels_dict = label_dict_construction()
            print(id_1)
            print(id_2)
            #L1 = list(graph_1.iloc[0]['node_labels']) if 'node_labels' in graph_1.columns else None
            #L1 = [label for label in L1 if label not in ['[', ']', ',', ' ', '"', "'"]]
            #L2 = list(graph_2.iloc[0]['node_labels']) if 'node_labels' in graph_2.columns else None
            #L2 = [label for label in L2 if label not in ['[', ']', ',', ' ', '"', "'"]]
            labels_1 = labels_dict.get(id_1, None)
            labels_2 = labels_dict.get(id_2, None)
           
            if labels_1 is not None and len(labels_1) != G_1.number_of_nodes():
                raise ValueError(f"Length of L1 ({len(L1)}) does not match the number of nodes in G_1 ({G_1.number_of_nodes()}).")
            if labels_2 is not None and len(labels_2) != G_2.number_of_nodes():
                raise ValueError(f"Length of L2 ({len(L2)}) does not match the number of nodes in G_2 ({G_2.number_of_nodes()}).")
            G_1, G_2, labels_1, labels_2 = compare_and_swap_graphs(G_1, G_2, labels_1, labels_2)
            print(labels_1)
            print(labels_2)
            label_distance_matrix = compute_label_distance(labels_1, labels_2, G_1.number_of_nodes())
            print(label_distance_matrix)
            structural_cross_matrix = compute_cross_matrix_with_structural_features(G_1, G_2)
            print(structural_cross_matrix)
            print('Cross Matrix without structural features')
            cross_matrix_no_structural_features = calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu=0.5, include_structural_features=False)
            print(cross_matrix_no_structural_features)
            print('GW score and Normalized GW score without structural features')
            gw_score, normalized_gw_score = compute_ged_GW(G_1, G_2, cross_matrix_no_structural_features)
            print(gw_score, normalized_gw_score)
            cross_matrix_with_structural_features = calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu=0.5, include_structural_features=True)
            print('Cross Matrix with structural features')
            print(cross_matrix_with_structural_features)
            gw_score, normalized_gw_score = compute_ged_GW(G_1, G_2, cross_matrix_with_structural_features)
            print('GW score and Normalized GW score without structural features')
            print(gw_score, normalized_gw_score)
            
        else:
            G_1, G_2 = compare_and_swap_graphs(G_1, G_2)
            print(f"G_1: {G_1.number_of_nodes()} nodes, {G_1.number_of_edges()} edges")
            print(f"G_2: {G_2.number_of_nodes()} nodes, {G_2.number_of_edges()} edges")
            label_distance_matrix = compute_label_distance(None, None, G_1.number_of_nodes())
            print("Labels distance matrix:")
            print(label_distance_matrix)
            print('Structural Cross Matrix')
            structural_cross_matrix = compute_cross_matrix_with_structural_features(G_1, G_2)
            print(structural_cross_matrix)
            print('Cross Matrix without structural features')
            cross_matrix_no_structural_features = calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu=0.5, include_structural_features=False)
            print(cross_matrix_no_structural_features)
            gw_score, normalized_gw_score = compute_ged_GW(G_1, G_2, cross_matrix_no_structural_features)
            print('GW score and Normalized GW score without structural features')
            print(gw_score, normalized_gw_score)            
            print('Cross Matrix with structural features')
            cross_matrix_with_structural_features = calculate_cross_matrix(label_distance_matrix, structural_cross_matrix, mu=0.5, include_structural_features=True)
            print(cross_matrix_with_structural_features)
            print('GW score and Normalized GW score with structural features')
            gw_score, normalized_gw_score = compute_ged_GW(G_1, G_2, cross_matrix_with_structural_features)
            print(gw_score, normalized_gw_score)
            
        
            
       
    
    




# for dataset in dataset_names:
#     dataset_path = os.path.join("Dataset", dataset, f"{dataset}_graphs.csv")
#     if os.path.exists(dataset_path):
#         print(f"Processing dataset: {dataset}")
#         graphs_df = pd.read_csv(dataset_path)
#         sampled_graphs_df = graphs_df.sample(n=3)
#         print(sampled_graphs_df)
#         load_and_process_dataframe(sampled_graphs_df, dataset)
        
#     else:
#         print(f"Dataset file not found: {dataset_path}")

#OCCORRE AGGIUNGERE LE DUMMY LABELS!

# for dataset in dataset_names:
#     if dataset != 'AIDS':
#         graphs = load_graphs(dataset)
#         for F in graphs.keys():
#             for S in graphs.keys():
#                 if F != S:
#                     G_F, G_S = graphs[F], graphs[S]
#                     G_F, G_S = compare_and_swap_graphs(G_F, G_S)
#         print(10*'*')
#     else:
#         graphs = load_graphs(dataset)
#         for F in graphs.keys():
#             for S in graphs.keys():
#                 if F != S:
#                     G_F, G_S = graphs[F][0], graphs[S][0]
#                     G_F, G_S = compare_and_swap_graphs(G_F, G_S)
#         print(10*'*')
      
