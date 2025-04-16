import json
from utils import iterate_get_graphs, load_ged
import networkx as nx
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
from ot.gromov import fused_gromov_wasserstein, fused_gromov_wasserstein2

class Graph:
    def __init__(self, graph_identifier: int, graph_data: dict):
        self.graph_identifier: int = graph_identifier
        self.graph_data: dict = graph_data.copy()
        self.number_of_nodes: int = self.graph_data["n"]
        self.number_of_edges: int = self.graph_data["m"]
        self.labels: list = self.graph_data["labels"]
        self.graph: nx.Graph = nx.Graph()
        self.graph.add_edges_from(self.graph_data["graph"])
        self.structural_feature: np.ndarray = self.generate_structural_feature_matrix()

    def generate_structural_feature_matrix(self) -> np.ndarray:
        num_nodes: int = self.graph.number_of_nodes()
        structural_feature: np.ndarray = np.zeros((num_nodes, 3))

        pagerank: dict = nx.pagerank(self.graph)
        clustering_coeff: dict = nx.clustering(self.graph)

        for i, node in enumerate(self.graph.nodes()):
            degree_centrality: float = nx.degree_centrality(self.graph)[node]
            pr: float = pagerank[node]
            cc: float = clustering_coeff[node]
            structural_feature[i] = [degree_centrality, pr, cc]

        return structural_feature

    def add_dummy_nodes(self, target_number_of_nodes: int) -> None:
        current_number_of_nodes: int = self.graph.number_of_nodes()
        if current_number_of_nodes < target_number_of_nodes:
            for i in range(current_number_of_nodes, target_number_of_nodes):
                self.graph.add_node(i)
        self.number_of_nodes = self.graph.number_of_nodes()
        self.structural_feature = self.generate_structural_feature_matrix()

class ComputeGED:
    def __init__(self, graph1: Graph, graph2: Graph, with_feature: bool = False) -> None:
        self.graph1: Graph = graph1
        self.graph2: Graph = graph2
        self.with_feature: bool = with_feature
        self.n: int = 0
        self.nu: np.ndarray = np.array([])
        self.mu: np.ndarray = np.array([])
        self.cost1: np.ndarray = np.array([])
        self.cost2: np.ndarray = np.array([])
        self.cross_cost: np.ndarray = np.array([])

    def ensure_smaller_graph_first(self) -> None:
        if self.graph1.number_of_nodes > self.graph2.number_of_nodes:
            self.graph1, self.graph2 = self.graph2, self.graph1

    def add_dummy_nodes_to_smaller_graph(self) -> None:
        self.ensure_smaller_graph_first()
        difference: int = self.graph2.number_of_nodes - self.graph1.number_of_nodes
        if difference > 0:
            self.graph1.add_dummy_nodes(self.graph1.number_of_nodes + difference)

    def set_cost(self) -> None:
        self.n = self.graph1.number_of_nodes
        self.nu = np.full(self.n, 1 / self.n)
        self.mu = np.full(self.n, 1 / self.n)
        #self.cost1 = nx.to_numpy_array(self.graph1.graph) - np.eye(self.n)
        #self.cost2 = nx.to_numpy_array(self.graph2.graph) - np.eye(self.n)
        #self.cost1 = nx.to_numpy_array(self.graph1.graph, nodelist=range(self.n)) - np.eye(self.n)
        #self.cost2 = nx.to_numpy_array(self.graph2.graph, nodelist=range(self.n)) - np.eye(self.n)

        self.cost1 = nx.to_numpy_array(self.graph1.graph, nodelist=range(self.n))
        self.cost2 = nx.to_numpy_array(self.graph2.graph, nodelist=range(self.n)) 
        
        self.cost1 *= self.n
        self.cost2 *= self.n

    def compute_cross_cost(self) -> None:
        if self.with_feature:
            self.cross_cost = np.linalg.norm(
                self.graph1.structural_feature[:, None, :] - self.graph2.structural_feature[None, :, :],
                axis=2
            )
        else:
            self.cross_cost = np.ones((self.n, self.n))

    def process(self) -> tuple[np.ndarray, float]:
        alpha_test: float = 1 / 3
        reverse: float = 3 / 2
        T_cg, log_cg = fused_gromov_wasserstein(
            self.cross_cost, self.cost1, self.cost2, self.mu, self.nu,
            loss_fun="square_loss", alpha=alpha_test, armijo=True, verbose=False, log=True
        )
        pre_ged: float = log_cg["fgw_dist"] * reverse
        return T_cg[:self.n, :], pre_ged

def load_graphs(dataset_name):
    train_graphs = iterate_get_graphs(f"json_data/{dataset_name}/train", "json")
    test_graphs = iterate_get_graphs(f"json_data/{dataset_name}/test", "json")
    return train_graphs + test_graphs

def get_graphs_identifiers(graphs):
    return [graph['gid'] for graph in graphs]

def prepare_test_set(dataset_name, ged_file, sample_size=10):
        graphs = load_graphs(dataset_name)
        gids = get_graphs_identifiers(graphs)
        _, ged_dict = load_ged(dataset_name, ged_file)
        sampled_keys = random.sample(list(ged_dict.keys()), sample_size)
        test_set = [(key, ged_dict[key]) for key in sampled_keys]
        #test_set = [(key, ged_dict[key]) for key in list(ged_dict.keys())]
        #print('Test Set: ', len(test_set))
        graph_dict = {graph['gid']: {k: v for k, v in graph.items() if k != 'gid'} for graph in graphs}
        return test_set, graph_dict


def compute_ged(Graph_1, Graph_2, with_features):
    T = ComputeGED(Graph_1, Graph_2, with_features)
    T.ensure_smaller_graph_first()
    T.add_dummy_nodes_to_smaller_graph()
    T.set_cost()
    T.compute_cross_cost()
    _, pre_ged = T.process()
    return pre_ged
  

        
        
        
   