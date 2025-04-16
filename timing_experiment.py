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
    sample_sizes: List[int] = list(np.linspace(50, 150000, 10, dtype=int))
    exec_time_with_no_features: List[float] = []
    exec_time_with_features: List[float] = []

    for sample_size in sample_sizes:
        test_set: List[Tuple[Tuple[int, int], float]] = prepare_test_set(dataset, "TaGED.json", sample_size=sample_size)[0]
        graph_dict: Dict[int, Dict] = prepare_test_set(dataset, "TaGED.json", sample_size=sample_size)[1]
        
        # Execution time without features
        start_time: float = time.time()
        for graph1_id, graph2_id in [(pair[0][0], pair[0][1]) for pair in test_set]:
            Graph_1: Graph = Graph(graph1_id, graph_dict[graph1_id])
            Graph_2: Graph = Graph(graph2_id, graph_dict[graph2_id])
            compute_ged(Graph_1, Graph_2, with_features=False)
        exec_time_with_no_features.append(time.time() - start_time)
        
        # Execution time with features
        start_time = time.time()
        for graph1_id, graph2_id in [(pair[0][0], pair[0][1]) for pair in test_set]:
            Graph_1 = Graph(graph1_id, graph_dict[graph1_id])
            Graph_2 = Graph(graph2_id, graph_dict[graph2_id])
            compute_ged(Graph_1, Graph_2, with_features=True)
        exec_time_with_features.append(time.time() - start_time)

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
    plt.xlabel('Sample Size')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs Sample Size')
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