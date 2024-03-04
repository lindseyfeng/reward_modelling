import torch
import json
import matplotlib.pyplot as plt
# Load logits from JSON
file = 'logits_scores_._open_llama_3b_rlhf_rm_without_2e-05__last_checkpoint_test.json~'
with open(file, 'r') as file:
    data = json.load(file)
logits_tensor = torch.tensor(data['logits'])

probabilities = torch.sigmoid(logits_tensor)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the distribution of logits on the first subplot
ax1.hist(logits_tensor.numpy(), bins=10, color='red', alpha=0.7)
ax1.set_title('Distribution of Logits')
ax1.set_xlabel('Logits')
ax1.set_ylabel('Frequency')

# Plotting the distribution of probabilities on the second subplot
ax2.hist(probabilities.numpy(), bins=10, color='blue', alpha=0.7)
ax2.set_title('Distribution of Probabilities')
ax2.set_xlabel('Probabilities')
ax2.set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()