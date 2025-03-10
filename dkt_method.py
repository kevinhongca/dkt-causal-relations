import torch
import json
import networkx as nx
import matplotlib.pyplot as plt

from pykt.models import init_model
from pykt.models.dkt import DKT  

# File paths
# Assistments 2009
# config_path = 'insert path to config.json'
# model_path = 'insert path to qid_model.ckpt'
# keyid2idx_path = 'insert path to keyid2idx.json'

# Assistments 2012
# config_path = 'insert path to config.json'
# model_path = 'insert path to qid_model.ckpt'
# keyid2idx_path = 'insert path to keyid2idx.json'

#Assistments 2017
# config_path = 'insert path to config.json'
# model_path = 'insert path to qid_model.ckpt'
# keyid2idx_path = 'insert path to keyid2idx.json'


#load model
with open(config_path, 'r') as file:
    config = json.load(file)
model_config = config['model_config']
data_config = config['data_config']

training_params = ['learning_rate', 'optimizer', 'batch_size', 'num_epochs', 'use_wandb','add_uuid']
model_config = {key: val for key, val in model_config.items() if key not in training_params}

model = init_model('dkt', model_config, data_config, 'qid')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

#load keyid2idx mapping
with open(keyid2idx_path, 'r') as file:
    keyid2idx = json.load(file)

keys = keyid2idx["concepts"].keys()
flipped_dict = {value: key for key, value in keyid2idx["concepts"].items()}
adjacency_matrix = []

#Jij
for concept_id in keys:
    question_ids = [concept_id]  
    responses = [1] 
    question_indices = [keyid2idx['concepts'][qid] for qid in question_ids]

    questions_tensor = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)
    responses_tensor = torch.tensor(responses, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        prediction = model(questions_tensor, responses_tensor)
        last_question_pred = prediction[:, -1, :]
        adjacency_matrix.append(last_question_pred.squeeze().tolist())


adjacency_matrix = torch.tensor(adjacency_matrix)

column_sums = adjacency_matrix.sum(dim=0, keepdim=True)
column_sums[column_sums == 0] = 1  #avoid division by 0
normalized_matrix = adjacency_matrix / column_sums

adjacency_list = {}
# threshold = 0.0107 #2009
# threshold = 0.0051 #2012
# threshold = 0.0139 #2017

for idx, concept_id in enumerate(keys):
    binary_connections = (normalized_matrix[idx] >= threshold).int().tolist()
    indices_of_ones = [flipped_dict[index] for index, value in enumerate(binary_connections) if value == 1 and index != keyid2idx['concepts'][concept_id]]
    adjacency_list[concept_id] = indices_of_ones

G = nx.DiGraph(adjacency_list)

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

# pos = nx.circular_layout(G)
# plt.figure(figsize=(12, 8)) 
# nx.draw(G, pos, with_labels=True, font_size=8, node_size=500, node_color='lightblue', edge_color='gray', alpha=0.5)
# plt.show()

if nx.is_directed_acyclic_graph(G):
    print("Graph is acyclic")
else:
    print("Graph is cyclic")

nodes_with_edges = [node for node, degree in G.out_degree() if degree >= 1]
print(nodes_with_edges)
print(len(nodes_with_edges))
