import os
import json
import torch
import networkx as nx
from node2vec import Node2Vec


BASE_DIR = 'json_data'                     
DATASETS = ['AIDS', 'IMDB', 'Linux']       
SUBDIRS = ['train', 'test']                

# Parametri Node2Vec
WALK_LENGTH = 20       
NUM_WALKS = 10         
P = 1.0                
Q = 1.0                
CONTEXT_SIZE = 10      
WORKERS = 4           
DIMENSIONS_LIST = [8, 16, 32]  # dimensioni degli embeddings

# Ciclo su dataset, dimensioni e sottocartelle
for dataset in DATASETS:
    for dim in DIMENSIONS_LIST:
        for subdir in SUBDIRS:
            # Directory di input
            input_dir = os.path.join(BASE_DIR, dataset, subdir)
            if not os.path.isdir(input_dir):
                print(f"Directory non trovata: {input_dir}, salto")
                continue

            # Directory di output direttamente nella cartella del dataset
            output_dir = os.path.join(BASE_DIR, dataset, f'embeddings_{dim}', subdir)
            os.makedirs(output_dir, exist_ok=True)

            # Elenco file JSON
            json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
            if not json_files:
                print(f"Nessun file .json in {input_dir}, salto")
                continue

            all_results = {}
            for fname in json_files:
                json_path = os.path.join(input_dir, fname)
                with open(json_path, 'r') as f:
                    data = json.load(f)

               
                G = nx.Graph()
                G.add_nodes_from(range(data['n']))
                G.add_edges_from(data['graph'])

    
                node2vec = Node2Vec(
                    graph=G,
                    dimensions=dim,
                    walk_length=WALK_LENGTH,
                    num_walks=NUM_WALKS,
                    p=P,
                    q=Q,
                    workers=WORKERS
                )
                model = node2vec.fit(window=CONTEXT_SIZE, min_count=1)

                # Prepara embedding tensor
                embeddings = torch.zeros(data['n'], dim)
                node_map = {}
                for node_id in range(data['n']):
                    vec = model.wv[str(node_id)]
                    embeddings[node_id] = torch.tensor(vec)
                    node_map[str(node_id)] = vec.tolist()

                # Salva file .pt
                base_name = os.path.splitext(fname)[0]
                out_pt = os.path.join(output_dir, f'embeddings_{base_name}.pt')
                torch.save(embeddings, out_pt)
                print(f'Salvato embedding PT per {dataset}/{subdir}/{fname} -> {out_pt}')

                # Aggiorna aggregati
                all_results[base_name] = node_map

            # Salva JSON aggregato
            out_json = os.path.join(output_dir, 'embeddings_all.json')
            with open(out_json, 'w') as jf:
                json.dump(all_results, jf, indent=2)
            print(f'Salvato JSON aggregato -> {out_json}')

