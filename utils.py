import json
import networkx as nx
import pandas as pd
import os

def build_ged_ground_truth_dataframe(dataset_name, file_name='TaGED.json'):
    path = 'json_data/' + dataset_name + '/' + file_name
    
    TaGED = json.load(open(path, 'r'))
    data = []
    for id_1, id_2, true_ged, ged_nc, ged_in, ged_ie, mappings in TaGED:
        data.append((id_1, id_2, true_ged))
    
    df = pd.DataFrame(data, columns=['id_1', 'id_2', 'true_ged'])
    print(df)
    print(10 * '*')
    
    output_folder = os.path.join("True_GED", dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{dataset_name}_ged.csv")
    df.to_csv(output_path, index=False)


def load_ged_ground_truth(dataset_name):
    file_path = os.path.join("True_GED", dataset_name, f"{dataset_name}_ged.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
        
    df = pd.read_csv(file_path)
    print(df.head(15))
    print(f"Shape of the dataframe: {df.shape}")

def label_dict_construction():
    label_dict = {}
    for type in ['train', 'test']:
        folder_path = os.path.join("json_data", "AIDS", type)
        try:
            files = [file for file in os.listdir(folder_path) if file.endswith('.onehot')]
            for file in files:
                graph_id = int(file.replace('.onehot', ''))
                with open(os.path.join(folder_path, file), 'r') as f:
                    content = json.load(f)
                    indexes = [sublist.index(1) for sublist in content]
                    label_dict[graph_id] = indexes
        except FileNotFoundError:
            print(f"File not found: {folder_path}")
    return label_dict

def save_graphs_on_df(dataset):
    if dataset == 'AIDS':
        graphs_df = pd.DataFrame(columns=['graph_id', 'graph_edge_list', 'node_labels'])
    else:
        graphs_df = pd.DataFrame(columns=['graph_id', 'graph_edge_list'])
    for type in ['train', 'test']:
        folder_path = os.path.join("json_data", dataset, type)
        try:
            files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
            for file in files:
                graph_id = int(file.replace('.json', ''))
                with open(os.path.join(folder_path, file), 'r') as f:
                    content = json.load(f)
                    if dataset != 'AIDS':
                        graphs_df = pd.concat([graphs_df, pd.DataFrame([{'graph_id': graph_id, 'graph_edge_list': content['graph']}])], ignore_index=True)
                    else:
                        node_labels = list([str(label) for label in content["labels"]])
                        graphs_df = pd.concat([graphs_df, pd.DataFrame([{'graph_id': graph_id, 'graph_edge_list': content['graph'], 'node_labels': node_labels}])], ignore_index=True)
        except FileNotFoundError:
            print(f"File not found: {folder_path}")
    
    output_folder = os.path.join("Dataset", dataset)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{dataset}_graphs.csv")
    graphs_df.to_csv(output_path, index=False)
    print(graphs_df.head(5))
    


def load_graphs(dataset):
    graphs = {}
    for type in ['train', 'test']:
        print(dataset, type)
        folder_path = os.path.join("json_data", dataset, type)
        print(folder_path)
        try:
            print(f"Dataset: {dataset}")
            files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
            first_ten_files = files[:2]
            print(first_ten_files)
            for file in first_ten_files:
                graph_id = int(file.replace('.json', ''))
                print(graph_id)
                graphs[graph_id] = []
                with open(os.path.join(folder_path, file), 'r') as f:
                    content = json.load(f)
                    edge_list = list(content['graph'])
                    G = nx.Graph()
                    G.add_edges_from(edge_list)
                    if dataset != 'AIDS':
                        graphs[graph_id] = G
                    else:
                        labels = content['labels']
                        graphs[graph_id] = (G, labels)
        except FileNotFoundError:
            print(f"File not found: {folder_path}")
    return graphs


    
    


