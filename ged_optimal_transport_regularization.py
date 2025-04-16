import json
from utils import iterate_get_graphs, load_ged
import networkx as nx
import numpy as np
import random
from scipy.optimize import linear_sum_assignment
from ot.gromov import fused_gromov_wasserstein, fused_gromov_wasserstein2

class Graph:
    def __init__(self, graph_identifier: int, graph_data: dict):
        self.graph_identifier = graph_identifier
        self.graph_data = graph_data.copy()
        self.number_of_nodes = self.graph_data["n"]
        self.number_of_edges = self.graph_data["m"]
        self.labels = self.graph_data["labels"]
        self.graph = nx.Graph()
        self.graph.add_edges_from(self.graph_data["graph"])
        self.structural_feature = self.generate_structural_feature_matrix()

   
    def generate_structural_feature_matrix(self):
        num_nodes = self.graph.number_of_nodes()
        structural_feature = np.zeros((num_nodes, 3))

        pagerank = nx.pagerank(self.graph)
        clustering_coeff = nx.clustering(self.graph)

        for i, node in enumerate(self.graph.nodes()):
            degree_centrality = nx.degree_centrality(self.graph)[node]
            pr = pagerank[node]
            cc = clustering_coeff[node]
            structural_feature[i] = [degree_centrality, pr, cc]
        
        return structural_feature
    
    def add_dummy_nodes(self, target_number_of_nodes):
        current_number_of_nodes = self.graph.number_of_nodes()
        if current_number_of_nodes < target_number_of_nodes:
            for i in range(current_number_of_nodes, target_number_of_nodes):
                self.graph.add_node(i)
        self.number_of_nodes = self.graph.number_of_nodes()
        self.structural_feature = self.generate_structural_feature_matrix()

class Compute_GED:
    def __init__(self, graph1: Graph, graph2: Graph, with_feature=False):
        self.graph1 = graph1
        self.graph2 = graph2
        self.with_feature = with_feature

    def ensure_smaller_graph_first(self):
        if self.graph1.number_of_nodes > self.graph2.number_of_nodes:
            self.graph1, self.graph2 = self.graph2, self.graph1

    def add_dummy_nodes_to_smaller_graph(self):
        self.ensure_smaller_graph_first()
        difference = self.graph2.number_of_nodes - self.graph1.number_of_nodes
        if difference > 0:
            self.graph1.add_dummy_nodes(self.graph1.number_of_nodes + difference)
    
    def set_cost(self):
        self.n = self.graph1.number_of_nodes
        self.nu = np.ones(self.n)
        self.mu = np.ones(self.n)
        self.cost1 = nx.adjacency_matrix(self.graph1.graph).toarray() - np.eye(self.graph1.number_of_nodes)
        self.cost2 = nx.adjacency_matrix(self.graph2.graph).toarray() - np.eye(self.graph2.number_of_nodes)
        self.nu = self.nu/self.n
        self.mu = self.mu/self.n
        self.cost1=self.cost1*self.n
        self.cost2 = self.cost2 * self.n
        
    def compute_cross_cost(self):
        self.cross_cost = np.zeros((self.n, self.n))        
        if self.with_feature:            
            for i in range(self.n):
                for j in range(self.n):
                    self.cross_cost[i, j] = np.linalg.norm(
                        self.graph1.structural_feature[i] - self.graph2.structural_feature[j]
                    )
        else:
            self.cross_cost = np.ones((self.n, self.n))
        

    def process(self):
        alpha_test = 1/3
        reverse = 3/2
        T_cg, log_cg = fused_gromov_wasserstein(
            self.cross_cost, self.cost1,self.cost2, self.mu, self.nu, 'square_loss',alpha=alpha_test,armijo=True,verbose=False, log=True)
        pre_ged = log_cg['fgw_dist']*reverse
        return T_cg[:self.n,:], pre_ged.item()

    # def compute_cost_matrix(self):
    #     num_nodes_g1 = self.graph1.number_of_nodes
    #     num_nodes_g2 = self.graph2.number_of_nodes
    #     cost_matrix = np.zeros((num_nodes_g1, num_nodes_g2))

    #     for i in range(num_nodes_g1):
    #         for j in range(num_nodes_g2):
    #             cost_matrix[i, j] = np.linalg.norm(
    #                         self.graph1.structural_feature[i] - self.graph2.structural_feature[j]
    #                     )
                
    #     return cost_matrix

    # def compute_ged(self):
    #     cost_matrix = self.compute_cost_matrix()
    #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #     ged = cost_matrix[row_ind, col_ind].sum()
    #     return ged

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
    T = Compute_GED(Graph_1, Graph_2, with_features)
    T.ensure_smaller_graph_first()
    T.add_dummy_nodes_to_smaller_graph()
    T.set_cost()
    T.compute_cross_cost()
    _, pre_ged = T.process()
    return pre_ged
  

        
        
        
   