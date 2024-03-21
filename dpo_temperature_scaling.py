import torch
from torch import optim, nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PhiForSequenceClassification
from datasets import load_dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import os

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
        print(confidences)
        print(accuracies)

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
            "prompt_input_ids":[]
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        chosen_str = prompt + " " + chosen
        rejected_str = prompt + " " + rejected
        tokenized_prompt = tokenizer(prompt)
        tokenized_chosen = tokenizer(chosen_str, padding = "max_length", max_length = 512)
        tokenized_rejected = tokenizer(rejected_str, padding = "max_length", max_length = 512)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["prompt_input_ids"].append(tokenized_prompt["input_ids"])

    return new_examples

def temperature_scale(logits, temperature):
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).cuda()
    return logits / temperature

def set_temperature_trl(valid_loader, model, temperature):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()
    with torch.no_grad():
        logits_list = []
        labels_list = []
        for inputs in valid_loader:
            # Stack and move to the correct device
            input_ids_chosen_tensor = inputs["input_ids_chosen"]
            attention_mask_chosen_tensor = inputs["attention_mask_chosen"]
            input_ids_rejected_tensor = inputs["input_ids_rejected"]
            attention_mask_rejected_tensor = inputs["attention_mask_rejected"]

            # Note: Corrected model input to use tensors instead of lists
            rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits

            # prob_pos_class = torch.sigmoid(rewards_chosen)
            # prob_neg_class = 1 - prob_pos_class
            # prob_chosen = torch.cat((prob_pos_class.unsqueeze(-1), prob_neg_class.unsqueeze(-1)), dim=-1)
            rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True).logits
            # prob_pos_class = torch.sigmoid(rewards_rejected)
            # prob_neg_class = 1 - prob_pos_class
            # prob_reject = torch.cat((prob_pos_class.unsqueeze(-1), prob_neg_class.unsqueeze(-1)), dim=-1)
            # Accumulate logits and labels
            pos_logits = rewards_chosen - rewards_rejected
            neg_logits = -pos_logits
            logits_list.append(torch.cat((pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)), dim=-1))
            # Convert logits list to tensor and labels list to tensor
        # llama3b
        logits = torch.cat(logits_list, dim=0).squeeze(1)  # This is your tensor from logits_list
        print(logits.shape)

        N, _ = logits.shape
        labels_list += [0] * N # Assuming binary labels, adjust as necessary
        labels = torch.tensor(labels_list).cuda()


        # print(labels)
            # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f ECE: %.3f' %  (before_temperature_nll, before_temperature_ece))

            # Optimize the temperature
        print(temperature.is_leaf) 
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(temperature_scale(logits, temperature), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

            # Calculate NLL after temperature scaling
        after_temperature_nll = nll_criterion(temperature_scale(logits, temperature), labels).item()
        after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
        print('Optimal temperature: %.3f' % temperature.item())
        print('After temperature - NLL: %.3f ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        return before_temperature_ece

def get_logps( logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)





def set_temperature(valid_loader, model, temperature):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()
    model.eval()
    with torch.no_grad():
        logits_list = []
        labels_list = []
        for inputs in valid_loader:
            # Stack and move to the correct device
            print("k")
            input_ids_chosen_tensor = torch.stack(inputs["input_ids_chosen"]).to(model.device).transpose(0, 1)
            attention_mask_chosen_tensor = torch.stack(inputs["attention_mask_chosen"]).to(model.device).transpose(0, 1)
            input_ids_rejected_tensor = torch.stack(inputs["input_ids_rejected"]).to(model.device).transpose(0, 1)
            attention_mask_rejected_tensor = torch.stack(inputs["attention_mask_rejected"]).to(model.device).transpose(0, 1)

            # Note: Corrected model input to use tensors instead of lists
            rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits
            ref_rewards_chosen = ref_file(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits

            rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True).logits
            ref_rewards_rejected = ref_file(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits
            # prob_pos_class = torch.sigmoid(rewards_rejected)
            # prob_neg_class = 1 - prob_pos_class
            # prob_reject = torch.cat((prob_pos_class.unsqueeze(-1), prob_neg_class.unsqueeze(-1)), dim=-1)
            # Accumulate logits and labels
            chosen_label = chosen_sequence_tokens["input_ids"][:]
            chosen_label[: len(inputs["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(inputs["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(inputs["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(inputs["prompt_input_ids"])

            chosen_logprob = get_logps(rewards_chosen, chosen_label)
            ref_chosen_logprob = get_logps(ref_rewards_chosen, chosen_label)
            reject_logprob = get_logps(rewards_reject, reject_label)
            ref_reject_logprob = get_logps(ref_rewards_reject, reject_label)

            pos_logits = (chosen_logprob-reject_logprob)-(ref_chosen_logprob-ref_reject_logprob)
            neg_logits = -pos_logits
            logits_list.append(torch.cat((pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)), dim=-1))
            # Convert logits list to tensor and labels list to tensor
        # llama3b
        logits = torch.cat(logits_list, dim=0).squeeze(1)  # This is your tensor from logits_list
        print(logits)
        N, _ = logits.shape
        labels_list += [0] * N # Assuming binary labels, adjust as necessary
        labels = torch.tensor(labels_list).cuda()

        # print(labels)

            # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f ECE: %.3f' %  (before_temperature_nll, before_temperature_ece))

            # Optimize the temperature
        print(temperature.is_leaf) 
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(temperature_scale(logits, temperature), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

            # Calculate NLL after temperature scaling
        after_temperature_nll = nll_criterion(temperature_scale(logits, temperature), labels).item()
        after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
        print('Optimal temperature: %.3f' % temperature.item())
        print('After temperature - NLL: %.3f ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        return before_temperature_ece

if __name__ == "__main__":
    ref_file = "../llama/llama-2-7b"
    model_file="dpo_llama7b_results/checkpoint-1000"
    print(model_file)
    tokenizer = AutoTokenizer.from_pretrained(file) #openlm-research/open_llama_3b
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(file).to(device)
    ref_model= AutoModelForCausalLM.from_pretrained(ref_file).to(device)
    raw_datasets = load_dataset("Dahoas/full-hh-rlhf")["test"].select(range(10))
    bsz = 5
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
    temperature = nn.Parameter((torch.ones(1)*1).to(device))
    print(temperature.is_leaf) 
    valid_loader = torch.utils.data.DataLoader(raw_datasets, pin_memory=True, batch_size=bsz, collate_fn=custom_collate_fn)
    print(valid_loader)
    set_temperature(valid_loader, model, temperature)
    