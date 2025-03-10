import torch
import json
import pandas as pd
import sys
import numpy as np
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

config_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2009_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_7eabe162-8d6b-4395-a505-043cfd54cc8d/config.json'
model_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/examples/saved_model/assist2009_dkt_qid_saved_model_42_0_0.2_200_0.001_1_1_7eabe162-8d6b-4395-a505-043cfd54cc8d/qid_model.ckpt'
keyid2idx_path = 'C:/Users/kevin/Documents/GitHub/pykt-toolkit-pt_emb/data/assist2009_fulldataset/keyid2idx.json'

with open(config_path, 'r') as file:
    config = json.load(file)
model_config = config['model_config']
data_config = config['data_config']
training_params = ['learning_rate', 'optimizer', 'batch_size', 'num_epochs', 'use_wandb', 'add_uuid']
model_config = {key: val for key, val in model_config.items() if key not in training_params}

model = init_model('dkt', model_config, data_config, 'qid')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

with open(keyid2idx_path, 'r') as file:
    keyid2idx = json.load(file)

flipped_dict = {value: key for key, value in keyid2idx["concepts"].items()}

#threshold (no longer needed, set as 1.0)
threshold = 1.0
max_no_improvement = 100  #T

target_concept_id = '278'  #desired exercise id
target_concept_idx = keyid2idx['concepts'][target_concept_id]

question_indices = [target_concept_idx]
questions_tensor = torch.tensor(question_indices, dtype=torch.long).unsqueeze(0)

responses = [1]
responses_tensor = torch.tensor(responses, dtype=torch.long).unsqueeze(0)

#initial understanding (dkt_method)
initial_understanding = None
with torch.no_grad():
    prediction = model(questions_tensor, responses_tensor)
    initial_understanding = prediction[0, -1, :].tolist()
    initial_understanding_rounded = [round(value, 3) for value in initial_understanding]

#top 3
top_3_initial = sorted([(idx, value) for idx, value in enumerate(initial_understanding_rounded) if idx != target_concept_idx], key=lambda x: x[1], reverse=True)[:3]
print("\nTop 3 Initial Execise Masteries:")
for idx, value in top_3_initial:
    print(f"Concept {flipped_dict.get(idx, idx)}: {value}")

predicted_understanding = 0.0
iteration_count = 0
no_improvement_count = 0
last_predicted_understanding = 0.0

while predicted_understanding < threshold:
    with torch.no_grad():
        prediction = model(questions_tensor, responses_tensor)
        last_question_pred = prediction[:, -1, :]
        predicted_understanding = last_question_pred[0, target_concept_idx].item()

    iteration_count += 1

    if predicted_understanding > last_predicted_understanding:
        last_predicted_understanding = predicted_understanding
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    # if iteration_count % 10 == 0:
    #     print(f"Iteration {iteration_count + 1}: Concept {target_concept_id}: {predicted_understanding:.3f}")

    if no_improvement_count >= max_no_improvement:
        # print(f"\nExiting loop.")
        break

    if predicted_understanding < threshold:
        responses.append(1)
        responses_tensor = torch.tensor(responses, dtype=torch.long).unsqueeze(0)

#final understanding (new_method)
final_understanding = None
with torch.no_grad():
    prediction = model(questions_tensor, responses_tensor)
    final_understanding = prediction[0, -1, :].tolist()
    final_understanding_rounded = [round(value, 3) for value in final_understanding]

#final understandings
top_3_final = sorted([(idx, value) for idx, value in enumerate(final_understanding_rounded) if idx != target_concept_idx], key=lambda x: x[1], reverse=True)[:3]
print("\nTop 3 Final Exercise Masteries:")
for idx, value in top_3_final:
    print(f"Concept {flipped_dict.get(idx, idx)}: {value}")
