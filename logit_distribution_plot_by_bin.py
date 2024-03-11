import torch
import json
import matplotlib.pyplot as plt
# Load logits from JSON
file = 'logits_scores_._open_llama_3b_rlhf_rm_bin_temperature5_2e-05_checkpoint-14000_test.json'
with open(file, 'r') as file:
    data = json.load(file)
logits_tensor = torch.tensor(data['logits'])

sigmoid_logits = torch.sigmoid(logits_tensor)
# correct_predictions = 1 == (sigmoid_logits[in_bin] >= 0.5).long()
# print(correct_predictions.float().mean().item())
sigmoid_neg_logits = torch.sigmoid(-logits_tensor)
labels_for_logits = torch.ones_like(sigmoid_logits).long()  # Labels for sigmoid_logits
labels_for_neg_logits = torch.zeros_like(sigmoid_neg_logits).long()  # Labels for sigmoid_neg_logits

probabilities = torch.cat((sigmoid_logits, sigmoid_neg_logits), dim=0)
labels = torch.cat((labels_for_logits, labels_for_neg_logits), dim=0)


# Define the bins for the confidence intervals
num_bins = 5
bin_boundaries = torch.linspace(0.5, 1, steps=num_bins + 1)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]
print(bin_lowers)
print(bin_uppers)
bin_centers = (bin_lowers + bin_uppers) / 2

# Initialize lists to store the accuracies and average confidences for each bin
bin_accuracies = []
bin_confidences = []
colors = ['blue', 'green', 'red', 'purple', 'orange']

# Recalculate in_bin_indices based on the first half of the probabilities to match the original logits_tensor size
for i, (bin_lower, bin_upper, color) in enumerate(zip(bin_lowers, bin_uppers, colors)):
    in_bin_indices = (sigmoid_logits >= bin_lower) & (sigmoid_logits < bin_upper)  # Adjusted to consider only sigmoid_logits
    logits_in_bin = logits_tensor[in_bin_indices]  # This should now correctly filter logits based on the bin

    # Plot with corrected logic
    ax.hist(logits_in_bin.numpy(), bins=30, color=color, alpha=0.6, label=f'Bin {i+1}: [{bin_lower.item(), bin_upper.item()})')

ax.set_title('Corrected Distribution of Logits by Probability Bins/testing data')
ax.set_xlabel('Logits')
ax.set_ylabel('Frequency')
ax.legend()

plt.tight_layout()
plt.grid(True)
plt.savefig('Corrected_Distribution_of_Logits_by_Probability_Bins.png')
