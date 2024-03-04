import torch
import json
import matplotlib.pyplot as plt
# Load logits from JSON
file = 'logits_scores_._open_llama_3b_rlhf_rm_without_2e-05__last_checkpoint_test.json~'
with open(file, 'r') as file:
    data = json.load(file)
logits_tensor = torch.tensor(data['logits'])

probabilities = torch.sigmoid(logits_tensor)

# Plotting
fig, ax1 = plt.subplots()

# Histogram for logits
color = 'tab:red'
ax1.set_xlabel('Value')
ax1.set_ylabel('Logits', color=color)
ax1.hist(logits_tensor.numpy(), bins=10, alpha=0.6, color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a second y-axis for the probabilities histogram
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Probabilities', color=color)  # we already handled the x-label with ax1
ax2.hist(probabilities.numpy(), bins=10, alpha=0.6, color=color)
ax2.tick_params(axis='y', labelcolor=color)

# Title and show
plt.title('Distribution of Logits and Corresponding Probabilities')
fig.tight_layout()  # To ensure there's no overlap
plt.grid(True)

# Save the figure
plt.savefig('distribution_without.png')
plt.close()