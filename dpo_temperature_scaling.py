import torch
from torch import optim, nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PhiForSequenceClassification, HfArgumentParser
from datasets import load_dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data.dataloader import default_collate

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    ref_file: Optional[str] = field(
        default="../llama/llama-2-7b",
        metadata={"help": "the location of the reference model name or path"},
    )
    model_file: Optional[str] = field(
        default="./dpo_llama7b_iterative_results/checkpoint-1000",
        metadata={"help": "the location of the SFT model name or path"},
    )
    bsz: Optional[int] = field(default=10, metadata={"help": "size of each batch"})


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
        softmaxes = torch.sigmoid(logits)
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
        percentage_true = torch.mean(accuracies.float()) * 100  
        print("accuracy : ", percentage_true.item())
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
            "ref_input_ids_chosen": [],
            "ref_attention_mask_chosen": [],
            "ref_input_ids_rejected": [],
            "ref_attention_mask_rejected": [],
            "prompt_length":[]
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        chosen_str = prompt + " " + chosen
        rejected_str = prompt + " " + rejected
        tokenized_chosen = ref_tokenizer(chosen_str, padding = "max_length", max_length = 512)
        tokenized_rejected = ref_tokenizer(rejected_str, padding = "max_length", max_length = 512)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["prompt_length"].append([len(prompt)])
        ref_tokenized_chosen = ref_tokenizer(chosen_str, padding = "max_length", max_length = 512)
        ref_tokenized_rejected = ref_tokenizer(rejected_str, padding = "max_length", max_length = 512)
        new_examples["ref_input_ids_chosen"].append(ref_tokenized_chosen["input_ids"])
        new_examples["ref_attention_mask_chosen"].append(ref_tokenized_chosen["attention_mask"])
        new_examples["ref_input_ids_rejected"].append(ref_tokenized_rejected["input_ids"])
        new_examples["ref_attention_mask_rejected"].append(ref_tokenized_rejected["attention_mask"])

    return new_examples

def temperature_scale(logits, temperature):
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).cuda()
    return logits / temperature

def set_temperature_trl(valid_loader, model, temperature):
    beta = 0.1
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()
    with torch.no_grad():
        logits_list = []
        labels_list = []
        for inputs in valid_loader:
            input_ids_chosen_tensor = inputs["chosen_input_ids"]
            attention_mask_chosen_tensor = inputs["chosen_attention_mask"]
            input_ids_rejected_tensor = inputs["rejected_input_ids"]
            attention_mask_rejected_tensor = inputs["rejected_attention_mask"]
            prompt_tensor = inputs["rejected_input_ids"]
            chosen_label = inputs["chosen_labels"]
            reject_label = inputs["rejected_labels"]
            rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits
            rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True).logits
            chosen_logprob = get_logps(rewards_chosen, chosen_label)
            reject_logprob = get_logps(rewards_rejected, reject_label)
            ref_chosen_logprob = inputs["reference_chosen_logps"]
            ref_reject_logprob = inputs["reference_rejected_logps"]
            pos_logits = ((chosen_logprob-ref_chosen_logprob)-(reject_logprob-ref_reject_logprob))*beta
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





def set_temperature(valid_loader, model, temperature, ref_model):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()
    model.eval()
    with torch.no_grad():
        logits_list = []
        labels_list = []
        beta = 0.1
        count = 0
        for inputs in valid_loader:
            print(count)
            count +=1
            # Stack and move to the correct device
            input_ids_chosen_tensor = torch.stack(inputs["input_ids_chosen"]).to(model.device).transpose(0, 1)
            attention_mask_chosen_tensor = torch.stack(inputs["attention_mask_chosen"]).to(model.device).transpose(0, 1)
            input_ids_rejected_tensor = torch.stack(inputs["input_ids_rejected"]).to(model.device).transpose(0, 1)
            attention_mask_rejected_tensor = torch.stack(inputs["attention_mask_rejected"]).to(model.device).transpose(0, 1)
            ref_input_ids_chosen_tensor = torch.stack(inputs["ref_input_ids_chosen"]).to(model.device).transpose(0, 1)
            ref_attention_mask_chosen_tensor = torch.stack(inputs["ref_attention_mask_chosen"]).to(model.device).transpose(0, 1)
            ref_input_ids_rejected_tensor = torch.stack(inputs["ref_input_ids_rejected"]).to(model.device).transpose(0, 1)
            ref_attention_mask_rejected_tensor = torch.stack(inputs["ref_attention_mask_rejected"]).to(model.device).transpose(0, 1)

            # Note: Corrected model input to use tensors instead of lists
            rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits
            ref_rewards_chosen = ref_model(input_ids=ref_input_ids_chosen_tensor, attention_mask=ref_attention_mask_chosen_tensor, return_dict=True).logits

            rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True).logits
            ref_rewards_rejected = ref_model(input_ids=ref_input_ids_rejected_tensor, attention_mask=ref_attention_mask_rejected_tensor, return_dict=True).logits
            label_pad_token_id = -100
            chosen_label = input_ids_chosen_tensor[:]
            reject_label = input_ids_rejected_tensor[:]
            for i, prompt_length in enumerate(inputs["prompt_length"][0]):
                prompt_length = prompt_length.item() 
                chosen_label[i, :prompt_length] = label_pad_token_id
                reject_label[i, :prompt_length] = label_pad_token_id

            ref_chosen_label = ref_input_ids_chosen_tensor[:]
            ref_reject_label = ref_input_ids_rejected_tensor[:]
            for i, prompt_length in enumerate(inputs["prompt_length"][0]):
                prompt_length = prompt_length.item() 
                ref_chosen_label[i, :prompt_length] = label_pad_token_id
                ref_reject_label[i, :prompt_length] = label_pad_token_id

            chosen_logprob = get_logps(rewards_chosen, chosen_label)
            ref_chosen_logprob = get_logps(ref_rewards_chosen, chosen_label)
            reject_logprob = get_logps(rewards_rejected, reject_label)
            ref_reject_logprob = get_logps(ref_rewards_rejected, reject_label)

            pos_logits = ((chosen_logprob-ref_chosen_logprob)-(reject_logprob-ref_reject_logprob))*beta
            neg_logits = -pos_logits
            logits_list.append(torch.cat((pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)), dim=-1))
            # Convert logits list to tensor and labels list to tensor
        # llama3b
        logits = torch.cat(logits_list, dim=0).squeeze(1)  # This is your tensor from logits_list
        N, _ = logits.shape
        labels_list += [0] * N # Assuming binary labels, adjust as necessary
        labels = torch.tensor(labels_list).cuda()

        # print(labels)

            # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f ECE: %.3f' %  (before_temperature_nll, before_temperature_ece))

            # Optimize the temperature
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
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    ref_file = script_args.ref_file
    model_file= script_args.model_file
    print(model_file)
    print(ref_file)
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_file) #openlm-research/open_llama_3b
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_file).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model= AutoModelForCausalLM.from_pretrained(ref_file).to(device)
    ref_model.config.pad_token_id = ref_tokenizer.pad_token_id
    raw_datasets = load_dataset("Dahoas/full-hh-rlhf")["test"]
    bsz = script_args.bsz
    print(bsz)
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
    valid_loader = torch.utils.data.DataLoader(raw_datasets, pin_memory=True, batch_size=bsz, collate_fn=custom_collate_fn)
    set_temperature(valid_loader, model, temperature, ref_model)
    
