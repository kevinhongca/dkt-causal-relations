import torch
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pykt.models import init_model

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
training_params = ['learning_rate', 'optimizer', 'batch_size', 'num_epochs', 'use_wandb', 'add_uuid']
model_config = {key: val for key, val in model_config.items() if key not in training_params}


model = init_model('dkt', model_config, data_config, 'qid')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

#load mapping
with open(keyid2idx_path, 'r') as file:
    keyid2idx = json.load(file)
flipped_dict = {value: key for key, value in keyid2idx['concepts'].items()}


# threshold = 0.0129 #2009
# threshold = 0.0067 2012 (?)
# threshold = 0.0167 2017
max_no_improvement = 100

adjacency_list = {}

for concept_id in keyid2idx['concepts']:
    question_indices = [keyid2idx['concepts'][concept_id]]
    questions_tensor = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)
    responses = [1]  
    responses_tensor = torch.tensor(responses, dtype=torch.long).unsqueeze(0)

    predicted_understanding = 0.0
    last_predicted_understanding = 0.0
    iteration_count = 0
    no_improvement_count = 0

    while no_improvement_count < max_no_improvement:
        with torch.no_grad():
            prediction = model(questions_tensor, responses_tensor)
            last_question_pred = prediction[:, -1, :]
            predicted_understanding = last_question_pred[0, question_indices[0]].item()

        if predicted_understanding > last_predicted_understanding:
            last_predicted_understanding = predicted_understanding
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        if no_improvement_count >= max_no_improvement:
            break

        responses.append(1)
        responses_tensor = torch.tensor(responses, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        prediction = model(questions_tensor, responses_tensor)
        last_question_pred = prediction[:, -1, :]
        total_sum = last_question_pred.sum().item()
        normalized_pred = last_question_pred / total_sum

    normalized_pred = torch.where(normalized_pred < threshold, torch.tensor(0), torch.tensor(1))
    normalized_pred_list = normalized_pred[0].tolist()
    connected_concepts = [flipped_dict[idx] for idx, value in enumerate(normalized_pred_list) if value == 1 and idx != question_indices[0]]
    adjacency_list[concept_id] = connected_concepts

G = nx.DiGraph(adjacency_list)

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

if nx.is_directed_acyclic_graph(G):
    print("Graph is acyclic")
else:
    print("Graph is cyclic")

nodes_with_degree_greater_than_one = [node for node, degree in G.degree() if degree >= 1]
print(nodes_with_degree_greater_than_one)
print(len(nodes_with_degree_greater_than_one))

