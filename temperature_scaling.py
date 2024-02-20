import torch
from torch import optim, nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data.sampler import SubsetRandomSampler

temperature = nn.Parameter(torch.ones(1) * 1.5)

from torch.utils.data.dataloader import default_collate

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
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, padding = "max_length", max_length = 512)
        tokenized_rejected = tokenizer(rejected, padding = "max_length", max_length = 512)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples

def temperature_scale(logits, temperature):
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

def set_temperature(valid_loader, model, temperature):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for inputs in valid_loader:
            # Stack and move to the correct device
            input_ids_chosen_tensor = torch.stack(inputs["input_ids_chosen"]).to(model.device)
            attention_mask_chosen_tensor = torch.stack(inputs["attention_mask_chosen"]).to(model.device)

            input_ids_rejected_tensor = torch.stack(inputs["input_ids_rejected"]).to(model.device)
            attention_mask_rejected_tensor = torch.stack(inputs["attention_mask_rejected"]).to(model.device)

            # Note: Corrected model input to use tensors instead of lists
            rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True)["logits"]
            rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True)["logits"]

            # Accumulate logits and labels
            logits_list.append(rewards_chosen)
            logits_list.append(rewards_rejected)
            labels_list += [1] * len(inputs["input_ids_chosen"])  # Assuming binary labels, adjust as necessary
            labels_list += [0] * len(inputs["input_ids_rejected"])

            # Convert logits list to tensor and labels list to tensor
            logits = torch.cat(logits_list).cuda()
            labels = torch.tensor(labels_list).cuda()

            # Calculate NLL and ECE before temperature scaling
            before_temperature_nll = nll_criterion(logits, labels).item()
            print('Before temperature - NLL: %.3f' % (before_temperature_nll))

            # Optimize the temperature
            optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(temperature_scale(logits, temperature), labels)
                loss.backward()
                return loss
            optimizer.step(eval)

            # Calculate NLL after temperature scaling
            after_temperature_nll = nll_criterion(temperature_scale(logits, temperature), labels).item()
            print('Optimal temperature: %.3f' % temperature.item())
            print('After temperature - NLL: %.3f' % (after_temperature_nll))


tokenizer = AutoTokenizer.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
PAD_TOKEN = '[PAD]'
if tokenizer.pad_token is None:
    tokenizer.pad_token = PAD_TOKEN
model = AutoModelForSequenceClassification.from_pretrained("weqweasdas/hh_rlhf_rm_open_llama_3b")
raw_datasets = load_dataset("Anthropic/hh-rlhf")["test"].shuffle(seed=42).select(range(2000))
raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,

    )
raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= 512
        and len(x["input_ids_rejected"]) <= 512
    )
print(raw_datasets)
valid_loader = torch.utils.data.DataLoader(raw_datasets, pin_memory=True, batch_size=32, collate_fn=custom_collate_fn)
print(valid_loader)
set_temperature(valid_loader, model, temperature)
    


    
