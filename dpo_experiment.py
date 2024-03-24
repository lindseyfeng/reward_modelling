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


def preprocess_function(examples):
    new_examples = {
            "chosen": [],
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "ref_input_ids_chosen": [],
            "ref_attention_mask_chosen": [],
            "ref_input_ids_rejected": [],
            "ref_attention_mask_rejected": [],
            "prompt_length":[],
            "ref_prompt_length":[]
    }
    for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
        chosen_str = prompt + " " + chosen
        rejected_str = prompt + " " + rejected
        tokenized_prompt = tokenizer(prompt)
        new_examples["chosen"].append(chosen_str)
        tokenized_chosen = tokenizer(chosen_str, padding = "max_length", max_length = 512)
        tokenized_rejected = tokenizer(rejected_str, padding = "max_length", max_length = 512)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["prompt_length"].append([len(tokenized_prompt)])
        ref_tokenized_prompt = ref_tokenizer(prompt)
        ref_tokenized_chosen = ref_tokenizer(chosen_str, padding = "max_length", max_length = 512)
        ref_tokenized_rejected = ref_tokenizer(rejected_str, padding = "max_length", max_length = 512)
        new_examples["ref_input_ids_chosen"].append(ref_tokenized_chosen["input_ids"])
        new_examples["ref_attention_mask_chosen"].append(ref_tokenized_chosen["attention_mask"])
        new_examples["ref_input_ids_rejected"].append(ref_tokenized_rejected["input_ids"])
        new_examples["ref_attention_mask_rejected"].append(ref_tokenized_rejected["attention_mask"])
        new_examples["ref_prompt_length"].append([len(ref_tokenized_prompt)])

    return new_examples

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    ref_file = script_args.ref_file
    pretrained_model_name_or_path = script_args.model_file
    print(pretrained_model_name_or_path)
    print(ref_file)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path) #openlm-research/open_llama_3b
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_file) #openlm-research/open_llama_3b
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    ref_model= AutoModelForCausalLM.from_pretrained(ref_file).to(device)
    ref_model.config.pad_token_id = ref_tokenizer.pad_token_id
    raw_datasets = load_dataset("Dahoas/full-hh-rlhf")["test"].select(range(10))
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
    beta = 0.1
    temperature = nn.Parameter((torch.ones(1)*1).to(device))
    valid_loader = torch.utils.data.DataLoader(raw_datasets, pin_memory=True, batch_size=bsz, collate_fn=custom_collate_fn)
    logits = []
    prompts = []
    count =0
    for inputs in valid_loader:
        count +=1
        print(count)
        input_ids_chosen_tensor = torch.stack(inputs["input_ids_chosen"]).to(model.device).transpose(0, 1)
        attention_mask_chosen_tensor = torch.stack(inputs["attention_mask_chosen"]).to(model.device).transpose(0, 1)
        input_ids_rejected_tensor = torch.stack(inputs["input_ids_rejected"]).to(model.device).transpose(0, 1)
        attention_mask_rejected_tensor = torch.stack(inputs["attention_mask_rejected"]).to(model.device).transpose(0, 1)
        ref_input_ids_chosen_tensor = torch.stack(inputs["ref_input_ids_chosen"]).to(model.device).transpose(0, 1)
        ref_attention_mask_chosen_tensor = torch.stack(inputs["ref_attention_mask_chosen"]).to(model.device).transpose(0, 1)
        ref_input_ids_rejected_tensor = torch.stack(inputs["ref_input_ids_rejected"]).to(model.device).transpose(0, 1)
        ref_attention_mask_rejected_tensor = torch.stack(inputs["ref_attention_mask_rejected"]).to(model.device).transpose(0, 1)
        with torch.no_grad():
            rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits
            ref_rewards_chosen = ref_model(input_ids=ref_input_ids_chosen_tensor, attention_mask=ref_attention_mask_chosen_tensor, return_dict=True).logits

            rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True).logits
            ref_rewards_rejected = ref_model(input_ids=ref_input_ids_rejected_tensor, attention_mask=ref_attention_mask_rejected_tensor, return_dict=True).logits
            label_pad_token_id = -100
            chosen_label = input_ids_chosen_tensor
            reject_label = input_ids_rejected_tensor
            for i, prompt_length in enumerate(inputs["prompt_length"][0]):
                prompt_length = prompt_length.item() 
                chosen_label[i, :prompt_length] = label_pad_token_id
                reject_label[i, :prompt_length] = label_pad_token_id
            ref_chosen_label = ref_input_ids_chosen_tensor
            ref_reject_label = ref_input_ids_rejected_tensor
            for i, prompt_length in enumerate(inputs["ref_prompt_length"][0]):
                prompt_length = prompt_length.item() 
                ref_chosen_label[i, :prompt_length] = label_pad_token_id
                ref_reject_label[i, :prompt_length] = label_pad_token_id
                

            chosen_logprob = get_logps(rewards_chosen, chosen_label)
            ref_chosen_logprob = get_logps(ref_rewards_chosen, ref_chosen_label)
            reject_logprob = get_logps(rewards_rejected, reject_label)
            ref_reject_logprob = get_logps(ref_rewards_rejected, ref_reject_label)
            
            pos_logits = ((chosen_logprob-ref_chosen_logprob)-(reject_logprob-ref_reject_logprob))*beta

        prompts.extend(inputs["chosen"])
        logits.extend(logits_to_list(pos_logits))

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