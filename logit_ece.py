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
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in self.bin_ranges:
            # Calculating |confidence - accuracy| in each bin based on logits
            in_bin = (logits > bin_lower) & (logits <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()  # This step may need adjustment based on how you define confidence with logits
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def main():
    file = 'logits_scores_._open_llama_3b_rlhf_rm_iterative_temperature_2e-05_last_checkpoint_test.json'
    with open(file, 'r') as file:
        data = json.load(file)
    logits_tensor = torch.tensor(data['logits'])
    print(logits_tensor.shape)
    logits = torch.cat((logits_tensor.unsqueeze(1), -logits_tensor.unsqueeze(1)), dim=1)
    print(logits)
    labels = torch.zeros_like(logits).long() 
    ece_loss = _ECELossLogitBins(n_bins=5)
    ece = ece_loss(logits, labels)
    print(f"Expected Calibration Error (ECE): {ece.item()}")

if __name__ == "__main__":
    main()