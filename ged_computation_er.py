import numpy as np
import matplotlib.pylab as pl
from ot.gromov import (
    fused_gromov_wasserstein,
    entropic_fused_gromov_wasserstein,
    BAPG_fused_gromov_wasserstein,
)
import networkx
from networkx.generators.community import stochastic_block_model as sbm
from time import time
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt





def modify_graph(G3, n):
    nodes = list(G3.nodes)
    edges_to_add = np.random.randint(0, n // 4)
    edges_to_remove = np.random.randint(0, n // 4)

    recently_added_edges = set()
    edges_added = 0
    while edges_added < edges_to_add:
        u, v = np.random.choice(nodes, size=2, replace=False)
        if not G3.has_edge(u, v):
            G3.add_edge(u, v)
            recently_added_edges.add((u, v))
            edges_added += 1
    edges_removed = 0
    while edges_removed < edges_to_remove:
        u, v = random.sample(list(G3.edges), 1)[0]
        if (u, v) not in recently_added_edges and (v, u) not in recently_added_edges:
            G3.remove_edge(u, v)
            edges_removed += 1

    return G3, edges_added, edges_removed



def generate_graphs(number_of_nodes, link_probability = 0.5):
    G2 = networkx.erdos_renyi_graph(number_of_nodes, link_probability)
    labels = np.random.choice([0, 1], size=number_of_nodes)
    for i, label in enumerate(labels):
        G2.nodes[i]['label'] = label
    G3 = G2.copy()
    G3, edges_added, edges_removed = modify_graph(G3, number_of_nodes)
    return G2, G3, edges_added + edges_removed

def compute_cross_cost_matrix_no_structural_features(G2, G3, N2, N3):
    F2 = np.zeros((N2, 1))
    for i in range(N2):
        F2[i, 0] = 1 if G2.nodes[i]['label'] == 1 else 0
    F3 = np.zeros((N3, 1))
    for i in range(N3):
        F3[i, 0] = 1 if G3.nodes[i]['label'] == 1 else 0
    

    # Compute pairwise euclidean distance between node features
    M = (F2**2).dot(np.ones((1, N3))) + np.ones((N2, 1)).dot((F3**2).T) - 2 * F2.dot(F3.T)
    return F2, F3, M

def compute_structural_features(graph):
    degree_centrality = np.array(list(networkx.degree_centrality(graph).values()))
    pagerank_centrality = np.array(list(networkx.pagerank(graph).values()))
    clustering_coefficient = np.array(list(networkx.clustering(graph).values()))
    return np.vstack((degree_centrality, pagerank_centrality, clustering_coefficient)).T

def compute_cross_matrix_with_structural_features(G2, G3, N2, N3):

    F2 = np.zeros((N2, 1))
    for i in range(N2):
        F2[i, 0] = 1 if G2.nodes[i]['label'] == 1 else 0
    
    S2 = compute_structural_features(G2)

    F3 = np.zeros((N3, 1))
    for i in range(N3):
        F3[i, 0] = 1 if G3.nodes[i]['label'] == 1 else 0
    

    S3 = compute_structural_features(G3)

    # Concatenate F2 and S2
    F2 = np.hstack((F2, S2))

    # Concatenate F3 and S3
    F3 = np.hstack((F3, S3))

    # Compute pairwise euclidean distance between node features
    M = np.sum(F2**2, axis=1, keepdims=True).dot(np.ones((1, F3.shape[0]))) + \
        np.ones((F2.shape[0], 1)).dot(np.sum(F3**2, axis=1, keepdims=True).T) - 2 * F2.dot(F3.T)
    
    return F2, F3, M

        

def compute_estimated_ged(T_cg, C2, C3):
    # Transform T_cg: largest element of each row is 1, others are 0
    T_cg_transformed = np.zeros_like(T_cg)
    for i in range(T_cg.shape[0]):
        max_index = np.argmax(T_cg[i])
        T_cg_transformed[i, max_index] = 1

    # Verify if T_cg_transformed is a permutation matrix
    is_permutation_matrix = (
        np.all(T_cg_transformed.sum(axis=1) == 1) and np.all(T_cg_transformed.sum(axis=0) == 1)
    )

    if is_permutation_matrix:
        # Multiply the transformed T_cg by C2 to get transformed C2
        transformed_C2 = T_cg_transformed.T @ C2 @ T_cg_transformed
        # Compute the absolute difference between transformed C2 and C3
        difference_matrix = np.abs(transformed_C2 - C3)
        # Compute the sum of all elements in the difference matrix and divide by 2
        estimated_ged = np.sum(difference_matrix) / 2
        return estimated_ged
    else:
        return None

def compute_ged(num_nodes, alpha=0.5, iterations=30, structural_features = False):
    ged_values = []
    estimated_ged_values = []

    h2 = np.ones(num_nodes) / num_nodes
    h3 = np.ones(num_nodes) / num_nodes

    for _ in range(iterations):
        if num_nodes < 8:
            G2, G3, _ = generate_graphs(num_nodes)
            ged = networkx.graph_edit_distance(G2, G3)
        else:
            G2, G3, app_ged = generate_graphs(num_nodes)
            ged = app_ged

        ged_values.append(ged)
        C2 = networkx.to_numpy_array(G2)
        C3 = networkx.to_numpy_array(G3)
        if structural_features:
            F1, F2, M = compute_cross_matrix_with_structural_features(G2, G3, num_nodes, num_nodes)
        else:
            F1, F2, M = compute_cross_cost_matrix_no_structural_features(G2, G3, num_nodes, num_nodes)
        
                # Conditional Gradient algorithm
        T_cg, log_cg = fused_gromov_wasserstein(
                    M, C2, C3, h2, h3, "square_loss", alpha=alpha, tol_rel=1e-9, verbose=False, log=True
        )

        estimated_ged = compute_estimated_ged(T_cg, C2, C3)
        if estimated_ged is not None:
            estimated_ged_values.append(estimated_ged)
        else:
            estimated_ged_values.append(None)

            # Filter out None values from the results
        filtered_ged_values = [g for g, e in zip(ged_values, estimated_ged_values) if e is not None]
        filtered_estimated_ged_values = [e for e in estimated_ged_values if e is not None]

        # Compute Mean Absolute Error (MAE)
        mae = np.mean(np.abs(np.array(filtered_ged_values) - np.array(filtered_estimated_ged_values)))
            # Compute accuracy
        accuracy = np.mean(np.abs(np.array(filtered_ged_values) - np.array(filtered_estimated_ged_values)) <= 1) * 100

        
        return mae, accuracy

def compute_mae_acc(num_nodes, alpha, iterations, structural_features, runs=15):
    mae_values = []
    accuracy_values = []

    for _ in range(runs):
        m, a = compute_ged(num_nodes=num_nodes, alpha=alpha, iterations=iterations, structural_features=structural_features)
        mae_values.append(m)
        accuracy_values.append(a)

    mae_avg = np.mean(mae_values)
    mae_std = np.std(mae_values)
    accuracy_avg = np.mean(accuracy_values)
    accuracy_std = np.std(accuracy_values)

    return mae_avg, mae_std, accuracy_avg, accuracy_std


def compute_ged_regression(M, n, p, structural_features):
    ged_values = []
    #estimated_ged_values = []
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
    

def GED_prediction_regression(file_name):
    # Open the DataFrame from the given file name
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return
    
    # Split the data into training and test sets (80% training, 20% test)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # First regression: 'Gromow_Wasserstein_Score_Normalized' as predictor, 'GED' as output
    X_train = train_df[['Gromow_Wasserstein_Score_Normalized']]
    y_train = train_df['GED']
    X_test = test_df[['Gromow_Wasserstein_Score_Normalized']]
    y_test = test_df['GED']

    model_ged = LinearRegression()
    model_ged.fit(X_train, y_train)

    # Predict GED for the test set
    y_pred_ged = model_ged.predict(X_test)

    # Compute Mean Absolute Error (MAE)
    mae_ged = mean_absolute_error(y_test, y_pred_ged)
    print(f"Mean Absolute Error (GED): {mae_ged:.4f}")

    # Compute Accuracy
    accuracy_ged = np.mean(np.abs(np.round(y_pred_ged) - y_test) <= 1) * 100
    print(f"Accuracy (GED): {accuracy_ged:.2f}%")


    # Second regression: 'Gromov-Wasserstein Score' as predictor, 'Normalized GED' as output
    X_train_normalized = train_df[['Gromov-Wasserstein Score']]
    y_train_normalized = train_df['Normalized GED']
    X_test_normalized = test_df[['Gromov-Wasserstein Score']]
    y_test_normalized = test_df['Normalized GED']

    model_normalized_ged = LinearRegression()
    model_normalized_ged.fit(X_train_normalized, y_train_normalized)

    # Predict Normalized GED for the test set
    y_pred_normalized_ged = model_normalized_ged.predict(X_test_normalized)
    # Compute Mean Absolute Error (MAE) for Normalized GED
    mae_normalized_ged = mean_absolute_error(y_test_normalized, y_pred_normalized_ged)
    print(f"Mean Absolute Error (Normalized GED): {mae_normalized_ged:.4f}")

    # Compute Accuracy for Normalized GED
    accuracy_normalized_ged = np.mean(np.abs(np.round(y_pred_normalized_ged) - y_test_normalized) <= 0.1) * 100
    print(f"Accuracy (Normalized GED): {accuracy_normalized_ged:.2f}%")
    

def partial_timing(G2, G3, num_nodes, alpha=0.5, structural_features=False):

    h2 = np.ones(num_nodes) / num_nodes
    h3 = np.ones(num_nodes) / num_nodes
    C2 = networkx.to_numpy_array(G2)
    C3 = networkx.to_numpy_array(G3)
       
    start_time = time()
    if structural_features:
        F1, F2, M = compute_cross_matrix_with_structural_features(G2, G3, num_nodes, num_nodes)
    else:
        F1, F2, M = compute_cross_cost_matrix_no_structural_features(G2, G3, num_nodes, num_nodes)
        
        # Conditional Gradient algorithm
    T_cg, log_cg = fused_gromov_wasserstein(
                M, C2, C3, h2, h3, "square_loss", alpha=alpha, tol_rel=1e-6, verbose=False, log=True
        )
    end_time = time()
    return end_time-start_time

def partial_timing_k_graphs(k, num_nodes, alpha=0.5, structural_features=False, n_iterations = 15):
    cumulative_times = []
    for _ in range(n_iterations):  # Repeat 10 times
        graphs = [networkx.erdos_renyi_graph(num_nodes, 0.5) for _ in range(k)]
        for graph in graphs:
            labels = np.random.choice([0, 1], size=num_nodes)
            for i, label in enumerate(labels):
                graph.nodes[i]['label'] = label

        transformed_graphs = [modify_graph(graph.copy(), num_nodes)[0] for graph in graphs]
        
        cumulative_time = 0.0
        for G2, G3 in zip(graphs, transformed_graphs):
            cumulative_time +=partial_timing(G2, G3, num_nodes, alpha, structural_features)
            cumulative_times.append(cumulative_time)
            
    return np.mean(cumulative_times)
            

print('Without Features')
mae_avg, mae_std, accuracy_avg, accuracy_std = compute_mae_acc(num_nodes=9, alpha=0.5, iterations=50, structural_features=False)
print(f"MAE - Average: {mae_avg}, Std: {mae_std}")
print(f"Accuracy - Average: {accuracy_avg:.2f}%, Std: {accuracy_std:.2f}%")
print('With Features')
mae_avg, mae_std, accuracy_avg, accuracy_std = compute_mae_acc(num_nodes=9, alpha=0.5, iterations=50, structural_features=True)
print(f"MAE - Average: {mae_avg}, Std: {mae_std}")
print(f"Accuracy - Average: {accuracy_avg:.2f}%, Std: {accuracy_std:.2f}%")

print(10*'*')

compute_ged_regression(1000, 9, 0.5, False)
compute_ged_regression(1000, 9, 0.5, True)
print('Without Features')
GED_prediction_regression('risultati/regression-ged_results_no_structural_features.csv')
print('With Features')
GED_prediction_regression('risultati/regression-ged_results_structural_features.csv')





# #k_values = np.linspace(100, 500, 10, dtype=int)
#
k_values = np.linspace(50, 1500, 6, dtype=int)
times_without_features = []
times_with_features = []

for k in k_values:
     time_without_features = partial_timing_k_graphs(k, 9, alpha=0.5, structural_features=False, n_iterations=4)
     time_with_features = partial_timing_k_graphs(k, 9, alpha=0.5, structural_features=True, n_iterations=4)
     times_without_features.append(time_without_features)
     times_with_features.append(time_with_features)

# # Plot the results
plt.clf()
plt.plot(k_values, times_without_features, label="Without Features", linestyle='--', linewidth=2, marker='o', markersize=8)
plt.plot(k_values, times_with_features, label="With Features", linestyle='-', linewidth=2, marker='o', markersize=8)
plt.xlabel("# Pair of Graphs", fontsize=13)
plt.ylabel("Average Cumulative Time (s)", fontsize=13)
#plt.title("Timing vs Number of Graphs", fontsize=14)
plt.legend()
plt.grid()
#plt.show()
# Save the plot to the "figure" folder
plt.savefig('figure/timing_vs_number_of_graphs_er.png')
