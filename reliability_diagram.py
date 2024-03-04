import torch
import json
import matplotlib.pyplot as plt
# Load logits from JSON
with open('logits_scores_._open_llama_3b_rlhf_rm__2e-05__temperature_last_checkpoint_test.json', 'r') as file:
    data = json.load(file)
logits_tensor = torch.tensor(data['logits'])


# Assuming `logits` are loaded from a file and `labels` are all zeros
probabilities = torch.sigmoid(logits_tensor).squeeze()
labels = torch.ones_like(probabilities, dtype=torch.long) 

# Define the bins for the confidence intervals
num_bins = 10
bin_boundaries = torch.linspace(0, 1, steps=num_bins + 1)
bin_lowers = bin_boundaries[:-1]
bin_uppers = bin_boundaries[1:]
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
        correct_predictions = labels[in_bin] == (probabilities[in_bin] >= 0.5).long()
        accuracy = correct_predictions.float().mean().item()
        bin_accuracies.append(accuracy)
        
        # Calculate the average confidence for the bin
        avg_confidence = probabilities[in_bin].mean().item()
        bin_confidences.append(avg_confidence)
    else:
        bin_accuracies.append(0)  # No samples in the bin
        bin_confidences.append((bin_lower + bin_upper) / 2)

print(bin_accuracies)
print(bin_confidences)
# Remove bins with no samples from the plot
valid_bins = [i for i, acc in enumerate(bin_accuracies) if acc > 0]
valid_bin_centers = [bin_centers[i].item() for i in valid_bins]
valid_bin_accuracies = [bin_accuracies[i] for i in valid_bins]

# Plotting the reliability diagram
plt.figure(figsize=(8, 6))
plt.plot(valid_bin_centers, valid_bin_accuracies, marker='o', linestyle='-', color='b', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

# Histogram for the distribution of predicted probabilities
# Note: You need to normalize your histogram (via the density parameter) to properly visualize it alongside the reliability plot
plt.hist(probabilities, bins=bin_boundaries, alpha=0.25, edgecolor='black', density=True, label='Histogram')

plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('Reliability Diagram with Histogram')
plt.legend(loc='best')
plt.grid(True)

# Save the figure
plt.savefig('reliability_diagram_with_histogram.png')
plt.close()
