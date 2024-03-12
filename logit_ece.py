import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class _ECELossLogitBins(nn.Module):
    """
    Calculates the Expected Calibration Error of a model using logit bins instead of softmax probability bins.
    This class divides the logit outputs into specified bins and calculates the ECE based on these bins.
    """  
    def __init__(self, n_bins=5):
        """
        n_bins (int): number of logit interval bins
        """
        super(_ECELossLogitBins, self).__init__()
        # Assuming equally spaced bins in the logit space. Adjust according to your needs.
        self.bin_ranges = [(i, i+1) for i in range(n_bins)]  # Custom bins based on logits

    def forward(self, logits, labels):
        # No softmax needed as we're working with logits directly
        confidences, predictions = torch.max(logits, 1)
        print(confidences)
        print(predictions)
        accuracies = predictions.eq(labels)
        print(accuracies)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in self.bin_ranges:
            # Calculating |confidence - accuracy| in each bin based on logits
            in_bin = (logits > bin_lower) & (logits <= bin_upper)
            in_bin_any = in_bin.any(dim=1)
            print(in_bin_any)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin_any].float().mean()
                avg_confidence_in_bin = confidences[in_bin_any].mean()  # This step may need adjustment based on how you define confidence with logits
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

def main():
    num_bins = 5
    file = 'logits_scores_._open_llama_3b_rlhf_rm_iterative_temperature_2e-05_last_checkpoint_test.json'
    with open(file, 'r') as file:
        data = json.load(file)
    logits_tensor = torch.tensor(data['logits'])
    print(logits_tensor)
    logits = torch.cat((logits_tensor.unsqueeze(1), -logits_tensor.unsqueeze(1)), dim=1)
    print(logits)
    labels = torch.zeros(logits.size(0), dtype=torch.long) 
    ece_loss = _ECELossLogitBins(n_bins=num_bins)
    ece = ece_loss(logits, labels)
    print(f"Expected Calibration Error (ECE): {ece.item()}")
    bin_boundaries = torch.linspace(0, 5, steps=num_bins + 1)  # Adjust the range as needed
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2

    bin_accuracies = []
    bin_avg_logits = []
    confidences, predictions = torch.max(logits, 1)
    print(confidences)
    print(predictions)
    accuracies = predictions.eq(labels)
    print(accuracies)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (logits >= bin_lower) & (logits < bin_upper)
        if in_bin.any():
            in_bin_any = in_bin.any(dim=1)
            correct_predictions = labels[in_bin_any] == True  # Assuming binary classification
            accuracy = correct_predictions.float().mean().item()
            bin_accuracies.append(accuracy)
            avg_logit = confidences[in_bin_any].mean().item()
            bin_avg_logits.append(avg_logit)
        else:
            # Handle bins with no samples
            bin_accuracies.append(0)
            bin_avg_logits.append((bin_lower + bin_upper) / 2)

    # Plotting the reliability diagram for logits
    plt.figure(figsize=(8, 6))
    plt.plot(bin_avg_logits, bin_accuracies, marker='o', linestyle='-', color='b', label='Model')
    plt.xlabel('Average Confidence')
    plt.ylabel('Accuracy')
    plt.title("Reliability Diagram based on Logits")
    plt.legend(loc='best')
    plt.grid(True)

    # Save the figure
    plt.savefig('logit_reliability_diagram.png')
    plt.show()


if __name__ == "__main__":
    main()
    