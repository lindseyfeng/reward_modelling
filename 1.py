import matplotlib.pyplot as plt
import numpy as np
import torch

# Example logits
logits = torch.tensor([0.7, -1.2, 0.3, -2.5, 1.0, 2.2, -0.4, 0.5, -1.5, 0.9])

# Convert logits to probabilities using the sigmoid function
probabilities = torch.sigmoid(logits)

# Creating the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plotting the distribution of logits on the first subplot
axs[0].hist(logits.numpy(), bins=10, color='red', alpha=0.7)
axs[0].set_title('Distribution of Logits')
axs[0].set_xlabel('Logits')
axs[0].set_ylabel('Frequency')

# Plotting the distribution of probabilities on the second subplot
axs[1].hist(probabilities.numpy(), bins=10, color='blue', alpha=0.7)
axs[1].set_title('Distribution of Probabilities')
axs[1].set_xlabel('Probabilities')
axs[1].set_ylabel('Frequency')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
