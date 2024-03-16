import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from datasets import concatenate_datasets
import json
from torch.utils.data.dataloader import default_collate
import numpy as np

def preprocess_function(examples):
    new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "chosen": []
    }
    for chosen, rejected, prompt in zip(examples["chosen"], examples["rejected"], examples["prompt"]):
        tokenized_chosen = tokenizer(prompt + " " + chosen, padding = "max_length", max_length = 512)
        tokenized_rejected = tokenizer(prompt + " " +rejected, padding = "max_length", max_length = 512)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["chosen"].append(chosen)
    return new_examples

# Process dataset to generate labels

# Save processed data to JSON
def save_to_json(file_path, data_to_save):
    with open(file_path, 'w') as json_file:
        json.dump(data_to_save, json_file)
    print(f"Data saved to {file_path}")

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


if __name__ == "__main__":
    BETA = 0.7
    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained("./open_llama_3b_rlhf_rm_without_2e-05__last_checkpoint")
    raw_datasets = load_dataset("Dahoas/full-hh-rlhf")
    model.eval()  # Ensure the model is in evaluation mode
    bsz = 100
    raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=1,
        )
    raw_datasets = raw_datasets.filter(
            lambda x: len(x["input_ids_chosen"]) <= 512
            and len(x["input_ids_rejected"]) <= 512
        )
    combined_dataset = concatenate_datasets([raw_datasets['train'], raw_datasets['test']])
    print(combined_dataset)
    valid_loader = torch.utils.data.DataLoader(combined_dataset, pin_memory=True, batch_size=bsz, collate_fn=custom_collate_fn)

    chosen_id = []
    label = []
    count = 0
    with torch.no_grad(): 
        for inputs in valid_loader: # Ensure no gradients are computed
            count += 1
            print(count)
            input_ids_chosen_tensor = torch.stack(inputs["input_ids_chosen"]).to(model.device).transpose(0, 1)
            attention_mask_chosen_tensor = torch.stack(inputs["attention_mask_chosen"]).to(model.device).transpose(0, 1)
            input_ids_rejected_tensor = torch.stack(inputs["input_ids_rejected"]).to(model.device).transpose(0, 1)
            attention_mask_rejected_tensor = torch.stack(inputs["attention_mask_rejected"]).to(model.device).transpose(0, 1)
            rewards_chosen = model(
                input_ids=input_ids_chosen_tensor,
                attention_mask=attention_mask_chosen_tensor,
                return_dict=True,
            )["logits"]
            rewards_rejected = model(
                input_ids=input_ids_rejected_tensor,
                attention_mask=attention_mask_rejected_tensor,
                return_dict=True,
            )["logits"]
            # Compute softmax probabilities for chosen over rejected items
            exp_logits_chosen = torch.exp(rewards_chosen)
            exp_logits_rejected = torch.exp(rewards_rejected)
            probs_chosen = exp_logits_chosen / (exp_logits_chosen + exp_logits_rejected)
            updated_label = (1 - BETA) * 1 + BETA * probs_chosen.squeeze().cpu().numpy() #change 1 to label
            chosen_id.extend(inputs["chosen"])
            label.extend(updated_label)

    print(chosen_id)
    data_to_save_converted = [float(item) if isinstance(item, np.float32) else item for item in label]

    data_to_save = {
        "chosen": chosen_id,
        "label": data_to_save_converted
    }
    save_to_json("processed_iterative_epoch1_data.json", data_to_save)
    




        