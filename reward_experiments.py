from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data.sampler import SubsetRandomSampler
import json
device = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data.dataloader import default_collate
import numpy as np
np.random.seed(42)

def logits_to_list(logits_tensor):
    logits_list = logits_tensor.detach().cpu().tolist()
    # Flatten the list since the original tensor has a shape of [10, 1]
    logits_list = [item for sublist in logits_list for item in sublist]
    return logits_list

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
        tokenized_chosen = tokenizer(chosen, padding = "max_length", max_length = 512)
        tokenized_rejected = tokenizer(rejected, padding = "max_length", max_length = 512)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

# Example usage
pretrained_model_name_or_path = './open_llama_3b_rlhf_rm_iterative_temperature_2e-05_last_checkpoint' 
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path).to(device)

raw_datasets = load_dataset("Dahoas/full-hh-rlhf")["test"]
bsz = 30
raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
raw_datasets = raw_datasets.filter(

        lambda x: len(x["input_ids_chosen"]) <= 512
        and len(x["input_ids_rejected"]) <= 512
    )
valid_loader = torch.utils.data.DataLoader(raw_datasets, pin_memory=True, batch_size=bsz, collate_fn=custom_collate_fn)
logits = []
score = []
prompts = []
count = 0
for batch in valid_loader:
    count +=1
    print(count)
    input_ids_chosen_tensor = torch.stack(batch["input_ids_chosen"]).to(model.device).transpose(0, 1)
    attention_mask_chosen_tensor = torch.stack(batch["attention_mask_chosen"]).to(model.device).transpose(0, 1)
    input_ids_rejected_tensor = torch.stack(batch["input_ids_rejected"]).to(model.device).transpose(0, 1)
    attention_mask_rejected_tensor = torch.stack(batch["attention_mask_rejected"]).to(model.device).transpose(0, 1)
    with torch.no_grad():
        rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits
        rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True).logits
        logits.extend(logits_to_list(rewards_chosen-rewards_rejected))
        prompts.extend([p + " " + c for p, c in zip(batch["prompt"], batch["chosen"])])


data_to_save = {
    "logits": logits,
    "prompts": prompts
}

# Specify the file path where you want to save the JSON file.
file_path = 'logits_scores_{}_{}.json'.format(pretrained_model_name_or_path.replace("/", "_"), "test")

# Writing the data to a JSON file.
with open(file_path, 'w') as json_file:
    json.dump(data_to_save, json_file)

print(f"Data saved to {file_path}")

    # Apply softmax to convert logits to probabilities
