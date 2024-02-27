from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data.sampler import SubsetRandomSampler
import json
device = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data.dataloader import default_collate

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
            "chosen":[]
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, padding = "max_length", max_length = 512)
        tokenized_rejected = tokenizer(rejected, padding = "max_length", max_length = 512)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["chosen"].append(examples["chosen"])

    return new_examples

class TemperatureScaledModel(torch.nn.Module):
    def __init__(self, pretrained_model_name_or_path, temperature=1.0):
        super(TemperatureScaledModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
        self.temperature = temperature

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the original model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Apply temperature scaling to logits
        logits = outputs.logits / self.temperature

        # If you're working with a model that outputs a single score, you might adjust it here
        # For classification models, you'd typically return the scaled logits for further processing (e.g., softmax)
        return logits

# Example usage
pretrained_model_name_or_path = 'weqweasdas/hh_rlhf_rm_open_llama_3b'  # Example model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
PAD_TOKEN = '[PAD]'
if tokenizer.pad_token is None:
    tokenizer.pad_token = PAD_TOKEN
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path).to(device)
temperature_scaled_model = TemperatureScaledModel(pretrained_model_name_or_path=pretrained_model_name_or_path, temperature=2.524).to(device)

raw_datasets = load_dataset("Anthropic/hh-rlhf")["test"].shuffle(seed=42).select(range(3000))
bsz = 10
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
promtps = []
for batch in valid_loader:
    input_ids_chosen_tensor = torch.stack(batch["input_ids_chosen"]).to(model.device).transpose(0, 1)
    attention_mask_chosen_tensor = torch.stack(batch["attention_mask_chosen"]).to(model.device).transpose(0, 1)
        # Forward pass through the temperature scaled model
    with torch.no_grad():
        blogits = temperature_scaled_model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor)
        logits.extend(logits_to_list(blogits))
        bscore = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True)["logits"]
        score.extend(logits_to_list(bscore))
        prompts.extend(batch["chosen"])
    print(logits)
    print(score)
    print(batch)


data_to_save = {
    "after": logits,
    "before": score
    "prompts": promtps
}

# Specify the file path where you want to save the JSON file.
file_path = 'logits_scores_{}_{}.json'.format(pretrained_model_name_or_path, 3000)

# Writing the data to a JSON file.
with open(file_path, 'w') as json_file:
    json.dump(data_to_save, json_file)

print(f"Data saved to {file_path}")

    # Apply softmax to convert logits to probabilities
