from ged_optimal_transport_regularization import *  
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
#import random
import time
import seaborn as sns
from typing import List, Dict, Union, Tuple



def compute_execution_time(dataset: str, destination: str) -> pd.DataFrame:
    sample_sizes: List[int] = list(np.linspace(10, 100, 5, dtype=int))
    exec_time_with_no_features: List[float] = []
    exec_time_with_features: List[float] = []

    test_set, graph_dict = prepare_test_set(dataset, "TaGED.json", 5000)



    for sample_size in sample_sizes:
        exec_time_no_features_runs = []
        exec_time_with_features_runs = []

        for _ in range(10):  # Repeat the computation 10 times
            #test_set, graph_dict = precomputed_data[sample_size]
            # Sample a subset of the test set for the current sample size
            sampled_test_set = random.sample(test_set, sample_size)
            # Create a dictionary to map graph IDs to their corresponding Graph objects
            graph_dict = {graph['gid']: {k: v for k, v in graph.items() if k != 'gid'} for graph in sampled_test_set}
            # Execution time without features
            start_time = time.time()
            for graph1_id, graph2_id in [(pair[0][0], pair[0][1]) for pair in test_set]:
                Graph_1 = Graph(graph1_id, graph_dict[graph1_id])
                Graph_2 = Graph(graph2_id, graph_dict[graph2_id])
                compute_ged(Graph_1, Graph_2, with_features=False)
            exec_time_no_features_runs.append(time.time() - start_time)

            # Execution time with features
            start_time = time.time()
            for graph1_id, graph2_id in [(pair[0][0], pair[0][1]) for pair in test_set]:
                Graph_1 = Graph(graph1_id, graph_dict[graph1_id])
                Graph_2 = Graph(graph2_id, graph_dict[graph2_id])
                compute_ged(Graph_1, Graph_2, with_features=True)
            exec_time_with_features_runs.append(time.time() - start_time)

        # Take the average over 10 runs
        exec_time_with_no_features.append(np.mean(exec_time_no_features_runs))
        exec_time_with_features.append(np.mean(exec_time_with_features_runs))
        

    # Create a DataFrame to store the results
    results_df: pd.DataFrame = pd.DataFrame({
        "sample_size": sample_sizes,
        "exec_time_with_no_features": exec_time_with_no_features,
        "exec_time_with_features": exec_time_with_features
    })

    # Save the DataFrame to the specified destination
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    results_df.to_csv(destination, index=False)

    return results_df


def plot_execution_times(execution_times: pd.DataFrame, output_fig: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='sample_size', y='exec_time_with_no_features', data=execution_times, 
                 label='No Features', marker='v', linestyle='--', markersize=9)
    sns.lineplot(x='sample_size', y='exec_time_with_features', data=execution_times, 
                 label='With Features', marker='*', linestyle='-', markersize=9)
    plt.xlabel('Sample Size', fontsize=14)
    plt.ylabel('Execution Time (s)', fontsize=14)
    plt.title('Execution Time vs Sample Size', fontsize=16)
    plt.grid(True)
    plt.legend()
   
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig + '.jpg', dpi=300)
    plt.close()




if __name__ == "__main__":
    # List of datasets to evaluate execution times for different graph datasets
    datasets = ['AIDS', 'IMDB', 'Linux']  

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        destination = os.path.join("risultati", f"execution-time-{dataset}.csv")
        result = compute_execution_time(dataset, destination=destination)
        print(result)
        plot_execution_times(result, f"figures/execution-times-{dataset}")