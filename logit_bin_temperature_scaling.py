import torch
from torch import optim, nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PhiForSequenceClassification
from datasets import load_dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data.dataloader import default_collate

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """  
    def __init__(self, n_bins=5):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0.5, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1].to(device)
        self.bin_uppers = bin_boundaries[1:].to(device)

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
    


def custom_collate_fn(batch):
    # This function assumes that each element in `batch` is a dictionary
    # with keys 'input_ids_chosen', 'attention_mask_chosen', 'input_ids_rejected', 'attention_mask_rejected'.
    # Modify as necessary to match your dataset structure.

    batched_data = {}
    for key in batch[0].keys():
        # Use default_collate to handle the usual collation logic, such as
        # converting lists of tensors into a single tensor.
        batched_data[key] = default_collate([d[key] for d in batch])
    return batched_data


def preprocess_function(examples):
    new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        chosen_str = prompt + " " + chosen
        rejected_str = prompt + " " + rejected
        tokenized_chosen = tokenizer(chosen_str, padding = "max_length", max_length = 512)
        tokenized_rejected = tokenizer(rejected_str, padding = "max_length", max_length = 512)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

def temperature_scale(logits, temperature):
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).cuda()
    return logits / temperature

def set_temperature_bin(valid_loader, model, temperature_list, bin_boundaries):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()
    model.eval()
    logits_list = [[] for _ in range(len(bin_boundaries) - 1)]
    print(logits_list)
    count = 0
    for inputs in valid_loader:
        count +=1
        print(count)
        input_ids_chosen_tensor = torch.stack(inputs["input_ids_chosen"]).to(model.device).transpose(0, 1)
        attention_mask_chosen_tensor = torch.stack(inputs["attention_mask_chosen"]).to(model.device).transpose(0, 1)
        input_ids_rejected_tensor = torch.stack(inputs["input_ids_rejected"]).to(model.device).transpose(0, 1)
        attention_mask_rejected_tensor = torch.stack(inputs["attention_mask_rejected"]).to(model.device).transpose(0, 1)
        rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits
        rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True).logits
        pos_logits = rewards_chosen - rewards_rejected
        neg_logits = -pos_logits
        prob = torch.sigmoid(pos_logits)
        idx = 0
        for bin_lower, bin_upper in bin_ranges:
            in_bin = ((pos_logits >= bin_lower) & (pos_logits < bin_upper)) | ((neg_logits >= bin_lower) & ( neg_logits< bin_upper))
            for i, row in enumerate(in_bin):
                if row.any():  # Checks if there is at least one True in the row
                    logits_list[idx].append(torch.cat((pos_logits[i].unsqueeze(-1), neg_logits[i].unsqueeze(-1)), dim=-1).detach())
            idx += 1
    print(logits_list)
    
    ind = 0
    for bin_lower, bin_upper in bin_ranges:
        labels_list = []
        print("range: ", bin_lower, bin_upper)

        if len(logits_list[ind]) == 0:
            temperature_list[ind] =1
            ind +=1
            print(ind)
            continue
        logits = torch.cat(logits_list[ind], dim=0).squeeze(1)
        N, _ = logits.shape
        labels_list += [0] * N # Assuming binary labels, adjust as necessary
        labels = torch.tensor(labels_list).cuda()
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f ECE: %.3f' %  (before_temperature_nll, before_temperature_ece))
                # Optimize the temperature
        print(temperature_list[ind].is_leaf) 
        optimizer = optim.LBFGS([temperature_list[ind]], lr=0.01, max_iter=100)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(temperature_scale(logits, temperature_list[ind]), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

                # Calculate NLL after temperature scaling
        after_temperature_nll = nll_criterion(temperature_scale(logits, temperature_list[ind]), labels).item()
        after_temperature_ece = ece_criterion(temperature_scale(logits, temperature_list[ind]), labels).item()
        print('Optimal temperature: %.3f' % temperature_list[ind].item())
        print('After temperature - NLL: %.3f ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        ind += 1



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b") #openlm-research/open_llama_3b
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained("./open_llama_3b_rlhf_rm_without_2e-05__last_checkpoint").to(device)
    raw_datasets = load_dataset("Dahoas/full-hh-rlhf")["test"].select(range(20))
    bsz = 1
    num_bins = 5
    bin_ranges = [(i, i+1) for i in range(num_bins)]
    raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=1,
        )
    raw_datasets = raw_datasets.filter(
            lambda x: len(x["input_ids_chosen"]) <= 512
            and len(x["input_ids_rejected"]) <= 512
        )
    print(raw_datasets)
    temperature_list = [nn.Parameter((torch.ones(1) * 1.5).to(device)) for _ in range(num_bins)]
    valid_loader = torch.utils.data.DataLoader(raw_datasets, pin_memory=True, batch_size=bsz, collate_fn=custom_collate_fn)
    set_temperature_bin(valid_loader, model, temperature_list, bin_ranges)
    print(temperature_list)