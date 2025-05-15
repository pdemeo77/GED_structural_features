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




N2 = 6  # 2 communities
N3 = 6  # 3 communities
p2 = [[1.0, 0.1], [0.1, 0.9]]
#p3 = [[1.0, 0.1, 0.0], [0.1, 0.95, 0.1], [0.0, 0.1, 0.9]]


def generate_graphs(N2, N3, p):
    G2 = sbm(seed=0, sizes=[N2 // 2, N2 // 2], p=p)
    G3 = G2.copy()
    
    nodes = list(G3.nodes)
    edges_to_add = np.random.randint(0, 4)
    edges_to_remove = np.random.randint(0, 4)
    print(f"Edges to add: {edges_to_add}, Edges to remove: {edges_to_remove}")

    recently_added_edges = set()
    edges_added = 0
    while edges_added < edges_to_add:
        u, v = np.random.choice(nodes, size=2, replace=False)
        if not G3.has_edge(u, v):
            print(f"Adding edge: ({u}, {v})")
            G3.add_edge(u, v)
            recently_added_edges.add((u, v))
            edges_added += 1

    edges_removed = 0
    while edges_removed < edges_to_remove:
        u, v = random.sample(list(G3.edges), 1)[0]
        if (u, v) not in recently_added_edges and (v, u) not in recently_added_edges:
            print(f"Removing edge: ({u}, {v})")
            G3.remove_edge(u, v)
            edges_removed += 1

    part_G2 = [G2.nodes[i]["block"] for i in range(N2)]
    part_G3 = [G3.nodes[i]["block"] for i in range(N3)]
    return G2, G3, part_G2, part_G3

def compute_cross_cost_matrix_no_structural_features(N2, N3, part_G2, part_G3):
    F2 = np.zeros((N2, 1))
    for i, c in enumerate(part_G2):
        F2[i, 0] = c

    F3 = np.zeros((N3, 1))
    for i, c in enumerate(part_G3):
        F3[i, 0] = c

    # Compute pairwise euclidean distance between node features
    M = (F2**2).dot(np.ones((1, N3))) + np.ones((N2, 1)).dot((F3**2).T) - 2 * F2.dot(F3.T)
    return F2, F3, M

def compute_structural_features(graph):
    degree_centrality = np.array(list(networkx.degree_centrality(graph).values()))
    pagerank_centrality = np.array(list(networkx.pagerank(graph).values()))
    clustering_coefficient = np.array(list(networkx.clustering(graph).values()))
    return np.vstack((degree_centrality, pagerank_centrality, clustering_coefficient)).T

def compute_cross_matrix_with_structural_features(G2, G3, part_G2, part_G3):
    F2 = np.zeros((len(part_G2), 1))
    for i, c in enumerate(part_G2):
        F2[i, 0] = c

    S2 = compute_structural_features(G2)

    F3 = np.zeros((len(part_G3), 1))
    for i, c in enumerate(part_G3):
        F3[i, 0] = c

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


G2, G3, part_G2, part_G3 = generate_graphs(N2, N3, p2)

# Calculate the Graph Edit Distance (GED) between G2 and G3
ged = networkx.graph_edit_distance(G2, G3)

print(f"Graph Edit Distance: {ged}")

C2 = networkx.to_numpy_array(G2)
C3 = networkx.to_numpy_array(G3)


F1, F2, M = compute_cross_cost_matrix_no_structural_features(N2, N3, part_G2, part_G3)



h2 = np.ones(C2.shape[0]) / C2.shape[0]
h3 = np.ones(C3.shape[0]) / C3.shape[0]


alpha = 0.5

# Conditional Gradient algorithm
print("Conditional Gradient \n")

T_cg, log_cg = fused_gromov_wasserstein(
    M, C2, C3, h2, h3, "square_loss", alpha=alpha, tol_rel=1e-9, verbose=False, log=True
)


estimated_ged = compute_estimated_ged(T_cg, C2, C3)
if estimated_ged is not None:
    print("Estimated Graph Edit Distance (GED):")
    print(estimated_ged)
else:
    print("The transport matrix is not a permutation matrix, cannot compute estimated GED.")







F1, F2, M = compute_cross_matrix_with_structural_features(G2, G3, part_G2, part_G3)
# print(F1)
# print(F2)
# print(M)

# h2 = np.ones(C2.shape[0]) / C2.shape[0]
# h3 = np.ones(C3.shape[0]) / C3.shape[0]


# alpha = 0.5


# Conditional Gradient algorithm
print("Conditional Gradient \n")

T_cg, log_cg = fused_gromov_wasserstein(
    M, C2, C3, h2, h3, "square_loss", alpha=alpha, tol_rel=1e-9, verbose=False, log=True
)

print(
    "Fused Gromov-Wasserstein distance estimated with Conditional Gradient solver: "
    + str((3.0/2)*log_cg["fgw_dist"].item())
)

estimated_ged = compute_estimated_ged(T_cg, C2, C3)
if estimated_ged is not None:
    print("Estimated Graph Edit Distance (GED):")
    print(estimated_ged)
else:
    print("The transport matrix is not a permutation matrix, cannot compute estimated GED.")