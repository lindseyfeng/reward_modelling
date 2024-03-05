import json
import matplotlib.pyplot as plt

# Function to load data from a JSON file
def load_data_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['logits'], data['prompts']

# Load logits and prompts from both files
logits1, prompts1 = load_data_from_json('logits_scores_._open_llama_3b_rlhf_rm_without_2e-05__last_checkpoint_train.json')
logits2, prompts2 = load_data_from_json('logits_scores_._open_llama_3b_rlhf_rm__2e-05__temperature_last_checkpoint_train.json')

# Create mappings from prompts to logits for each file
prompt_to_logits1 = {prompt: logit for prompt, logit in zip(prompts1, logits1)}
prompt_to_logits2 = {prompt: logit for prompt, logit in zip(prompts2, logits2)}

# Match prompts and collect corresponding logits
matched_logits1 = []
matched_logits2 = []
for prompt in prompts1:
    if prompt in prompt_to_logits2:  # Ensure the prompt exists in both files
        matched_logits1.append(prompt_to_logits1[prompt])
        matched_logits2.append(prompt_to_logits2[prompt])

# Plotting
plt.scatter(matched_logits1, matched_logits2)
plt.xlabel('Logits from OpenLlama3b')
plt.ylabel('Logits from OpenLlama3b with Temperature')
plt.title('Comparing Logits for Matching Prompts')
plt.grid(True)
plt.axis('equal')
plt.savefig('logit_transformation.png')
