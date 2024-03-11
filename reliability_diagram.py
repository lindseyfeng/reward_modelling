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

# Calculate the accuracies and confidences for each bin
for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
    # Find the indices of probabilities that fall into the current bin
    in_bin = (probabilities >= bin_lower) & (probabilities < bin_upper)
    if in_bin.any():
        # Calculate the accuracy for the bin
        correct_predictions = labels[in_bin] == 1
        accuracy = correct_predictions.float().mean().item()
        bin_accuracies.append(accuracy)
        
        # Calculate the average confidence for the bin
        avg_confidence = probabilities[in_bin].mean().item()
        bin_confidences.append(avg_confidence)
    else:
        bin_accuracies.append(0)  # No samples in the bin
        bin_confidences.append((bin_lower + bin_upper) / 2)

print(bin_accuracies)
# Remove bins with no samples from the plot
valid_bins = [i for i, acc in enumerate(bin_accuracies) if acc > 0]
valid_bin_centers = [bin_confidences[i] for i in valid_bins]
valid_bin_accuracies = [bin_accuracies[i] for i in valid_bins]

# Plotting the reliability diagram
plt.figure(figsize=(8, 6))
plt.plot(valid_bin_centers, valid_bin_accuracies, marker='o', linestyle='-', color='b', label='Model')
plt.plot([0.5, 1], [0.5, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title("Reliability Diagram for OpenLlama 3B with temperature/testing data")
plt.legend(loc='best')
plt.grid(True)

# Save the figure
plt.savefig('reliability_diagram.png')
plt.close()
