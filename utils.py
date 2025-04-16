import os
import random
import json
from os.path import basename, isfile
from os import makedirs
from glob import glob
import networkx as nx

def list_and_print_random_json_file(source_folder):
    files = os.listdir(source_folder)
    json_files = [file for file in files if file.endswith('.json')]
    
    
    if not json_files:
        print("No JSON files found in the source folder.")
        return None
    
    random_file = random.choice(json_files)
    random_file_path = os.path.join(source_folder, random_file)
    
    with open(random_file_path, 'r') as file:
        content = json.load(file)
        #print(json.dumps(content, indent=4))
        return content

 
#def load_ged(ged_dict, data_location='', dataset_name='AIDS', file_name='TaGED.json'):
def load_ged(dataset_name='AIDS', file_name='TaGED.json'):
    '''
    list(tuple)
    ged = [(id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, [best_node_mapping])]

    id_1 and id_2 are the IDs of a graph pair, e.g., the ID of 4.json is 4.
    The given graph pairs satisfy that n1 <= n2.

    ged_value = ged_nc + ged_in + ged_ie
    (ged_nc, ged_in, ged_ie) is the type-aware ged following the setting of TaGSim.
    ged_nc: the number of node relabeling
    ged_in: the number of node insertions/deletions
    ged_ie: the number of edge insertions/deletions

    [best_node_mapping] contains 10 best matching at most.
    best_node_mapping is a list of length n1: u in g1 -> best_node_mapping[u] in g2

    return dict()
    ged_dict[(id_1, id_2)] = ((ged_value, ged_nc, ged_in, ged_ie), best_node_mapping_list)
    '''
    #path = "{}json_data/{}/{}".format(data_location, dataset_name, file_name)
    ged_dict = {}
    ged_dict_short = {}
    path = 'json_data/' + dataset_name + '/' + file_name
    TaGED = json.load(open(path, 'r'))
    for (id_1, id_2, ged_value, ged_nc, ged_in, ged_ie, mappings) in TaGED:
        ta_ged = (ged_value, ged_nc, ged_in, ged_ie)
        ged_dict[(id_1, id_2)] = (ta_ged, mappings)
        ged_dict_short[(id_1, id_2)] = ged_value
    return ged_dict, ged_dict_short

def sorted_nicely(l):
    """
    Sort file names in a fancy way.
    The numbers in file names are extracted and converted from str into int first,
    so file names can be sorted based on int comparison.
    :param l: A list of file names:str.
    :return: A nicely sorted file name list.
    """

    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_file_paths(dir, file_format='json'):
    """
    Return all file paths with file_format under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    dir = dir.rstrip('/')
    paths = sorted_nicely(glob(dir + '/*.' + file_format))
    return paths

def iterate_get_graphs(dir, file_format):
    """
    Read networkx (dict) graphs from all .gexf (.json) files under dir.
    :param dir: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ['gexf', 'json', 'onehot', 'anchor']
    graphs = []
    
    for file in get_file_paths(dir, file_format):
  
        gid = int(basename(file).split('.')[0])
        if file_format == 'gexf':
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        elif file_format == 'json':
            # g is a dict
            g = json.load(open(file, 'r'))
            g['gid'] = gid
        elif file_format in ['onehot', 'anchor']:
            # g is a list of onehot labels
            g = json.load(open(file, 'r'))
        graphs.append(g)
 
    return graphs

