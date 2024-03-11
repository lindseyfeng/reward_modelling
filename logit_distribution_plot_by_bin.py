import torch
import json
import matplotlib.pyplot as plt

# Load logits from JSON
file = 'logits_scores_._open_llama_3b_rlhf_rm_iterative_temperature_2e-05_last_checkpoint_test.json'
with open(file, 'r') as file:
    data = json.load(file)
logits_tensor = torch.tensor(data['logits'])

# Compute the sigmoid of the logits
sigmoid_logits = torch.sigmoid(logits_tensor)
sigmoid_neg_logits = torch.sigmoid(-logits_tensor)

# Combine positive and negative sigmoid logits for a comprehensive probability distribution
probabilities = torch.cat((sigmoid_logits, sigmoid_neg_logits), dim=0)

# Define the bins for the confidence intervals
num_bins = 5
bin_boundaries = torch.linspace(0.5, 1, steps=num_bins + 1)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]

# Colors for each bin
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Initializing the figure and axes for plotting
fig, ax = plt.subplots(figsize=(10, 6))  # This defines 'ax'

# Plot the distribution of logits by probability bins
for i, (bin_lower, bin_upper, color) in enumerate(zip(bin_lowers, bin_uppers, colors)):
    in_bin_indices = (sigmoid_logits >= bin_lower) & (sigmoid_logits < bin_upper)
    logits_in_bin = logits_tensor[in_bin_indices]
    num_elements = logits_in_bin.shape[0]
    bin_lower_rounded = round(bin_lower.item(), 2)
    bin_upper_rounded = round(bin_upper.item(), 2)

    # Plotting
    ax.hist(logits_in_bin.numpy(), bins=30, color=color, alpha=0.6, label=f'Bin {i+1}: [{bin_lower.item():.2f}, {bin_upper.item():.2f}] (n={num_elements})')

ax.set_title('Corrected Distribution of Logits by Probability Bins/testing data')
ax.set_xlabel('Logits')
ax.set_ylabel('Frequency')
ax.legend()

plt.tight_layout()
plt.grid(True)
plt.savefig('Corrected_Distribution_of_Logits_by_Probability_Bins.png')
