import torch
import json
import matplotlib.pyplot as plt
# Load logits from JSON
file = 'logits_scores_._open_llama_3b_rlhf_rm__2e-05__temperature_last_checkpoint_test.json'
with open(file, 'r') as file:
    data = json.load(file)
logits_tensor = torch.tensor(data['logits'])

probabilities = torch.sigmoid(logits_tensor)

# Creating the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the distribution of logits on the first subplot
axs[0].hist(logits_tensor.numpy(), bins=10, color='red', alpha=0.7)
axs[0].set_title('Distribution of Logits with Temperature')
axs[0].set_xlabel('Logits')
axs[0].set_ylabel('Frequency')

# Plotting the distribution of probabilities on the second subplot
axs[1].hist(probabilities.numpy(), bins=10, color='blue', alpha=0.7)
axs[1].set_title('Distribution of Probabilities with Temperature')
axs[1].set_xlabel('Probabilities')
axs[1].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

plt.savefig('without_logits_and_probabilities_histograms.png')