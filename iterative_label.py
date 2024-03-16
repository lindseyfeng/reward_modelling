import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from datasets import concatenate_datasets
import json

def preprocess_function(examples):
    new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
    }
    for chosen, rejected, prompt in zip(examples["chosen"], examples["rejected"], examples["prompt"]):
        tokenized_chosen = tokenizer(prompt + " " + chosen, return_tensors='pt')
        tokenized_rejected = tokenizer(prompt + " " +rejected, return_tensors='pt')
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"][0])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"][0])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"][0])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"][0])
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
    valid_loader = torch.utils.data.DataLoader(combined_dataset, pin_memory=True, batch_size=bsz, collate_fn=custom_collate_fn)

    chosen_id = []
    label = []

    with torch.no_grad(): 
        for inputs in valid_loader: # Ensure no gradients are computed
            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"],
                return_dict=True,
            )["logits"]
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"],
                return_dict=True,
            )["logits"]
            # Compute softmax probabilities for chosen over rejected items
            exp_logits_chosen = torch.exp(rewards_chosen)
            exp_logits_rejected = torch.exp(rewards_rejected)
            probs_chosen = exp_logits_chosen / (exp_logits_chosen + exp_logits_rejected)
            updated_label = (1 - BETA) * 1 + BETA * probs_chosen.squeeze().cpu().numpy() #change 1 to label
            chosen_id += inputs["input_ids_chosen"]
            label += updated_label
        
    data_to_save = {
        "input_ids_chosen": chosen_id,
        "labels": label
    }
    save_to_json("processed_iterative_epoch1_data.json", data_to_save)
    




        