from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import RewardConfig
from reward_trainer import IterativeRewardTrainer #LabelCallback
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from typing import Any, Dict, List, Optional, Union
import torch
import wandb
import json

BETA = 0.7
ALPHA = 2e-5
TEMPERATURE = 1
EPOCH = 1

PAD_TOKEN = '[PAD]'
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = AutoModelForSequenceClassification.from_pretrained(
    "openlm-research/open_llama_3b", torch_dtype=torch.bfloat16, num_labels=1,
)
# Assuming `tokenizer` is your tokenizer instance
if tokenizer.pad_token is None:
    tokenizer.pad_token = PAD_TOKEN


raw_datasets = load_dataset("Dahoas/full-hh-rlhf")
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

reward_config = RewardConfig(
    do_eval = True,
    report_to="wandb",
    output_dir="iterative_baseline_1",
    learning_rate= ALPHA,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.001,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    deepspeed=None,
    local_rank=-1,
    remove_unused_columns=False,
    label_names=[],
    bf16= True,
    logging_strategy="steps",
    logging_steps=10,
    optim="adamw_hf",
    lr_scheduler_type="linear",
    max_length = 512
)

class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, padding: Union[bool, str] = True, 
                 max_length: Optional[int] = None, pad_to_multiple_of: Optional[int] = None, 
                 return_tensors: str = "pt"):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_chosen = []
        features_rejected = []
        margin = []
        labels = [] 
            # check if we have a margin. If we do, we need to batch it as well
        has_margin = "margin" in features[0]
        for feature in features:
                # check if the keys are named as expected
            if (
                "input_ids_chosen" not in feature
                or "input_ids_rejected" not in feature
                or "attention_mask_chosen" not in feature
                or "attention_mask_rejected" not in feature
            ):
                raise ValueError(
                    "The features should include `input_ids_chosen`, `attention_mask_chosen`, `input_ids_rejected` and `attention_mask_rejected`"
                )

            features_chosen.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                    }
                )
            features_rejected.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"],
                    }
                )
            if has_margin:
                margin.append(feature["margin"])
            
            if "label" in feature:  # Collect labels if present
                    labels.append(feature["label"])
                    
        batch_chosen = self.tokenizer.pad(
                features_chosen,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        batch_rejected = self.tokenizer.pad(
                features_rejected,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
        batch = {
                "input_ids_chosen": batch_chosen["input_ids"],
                "attention_mask_chosen": batch_chosen["attention_mask"],
                "input_ids_rejected": batch_rejected["input_ids"],
                "attention_mask_rejected": batch_rejected["attention_mask"],
                "return_loss": True,
            }
        if labels:  # Add labels to the batch if they were collected
                batch["label"] = torch.tensor(labels, dtype=torch.long)
        if has_margin:
                margin = torch.tensor(margin, dtype=torch.float)
                batch["margin"] = margin
        return batch


def preprocess_function(examples):
    new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "label": []
    }
    # file = 'logits_scores_._open_llama_3b_rlhf_rm_without_2e-05__last_checkpoint_train.json'
    # with open(file, 'r') as file:
    #     data = json.load(file)
    label = 1
    for chosen, rejected, prompt in zip(examples["chosen"], examples["rejected"], examples["prompt"]):
        tokenized_chosen = tokenizer(prompt + " " + chosen, padding = "max_length", max_length = reward_config.max_length)
        tokenized_rejected = tokenizer(prompt + " " +rejected, padding = "max_length", max_length = reward_config.max_length)
        # for l, c in zip(data[label], data[input_ids_chosen]):
        #     if c == input_ids_chosen:
        #         label = l
        # if label == -100:
        #     print("no label found!!!")
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        with torch.no_grad():  # Ensure no gradients are computed
            rewards_chosen = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen, return_dict=True)["logits"]
            rewards_rejected = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected, return_dict=True)["logits"]

            # Compute softmax probabilities for chosen over rejected items
        exp_logits_chosen = torch.exp(rewards_chosen)
        exp_logits_rejected = torch.exp(rewards_rejected)
        probs_chosen = exp_logits_chosen / (exp_logits_chosen + exp_logits_rejected)

            # Calculate the updated label based on some logic; you might need to adjust this
        updated_label = (label - BETA) * inputs["label"] + BETA * probs_chosen.squeeze().cpu().numpy()
        new_examples["label"].append(updated_label)
    return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
data_to_save = {
    "input_ids_chosen": raw_datasets["train"]["input_ids_chosen"],
    "label": raw_datasets["train"]["label"]
}
print(data_to_save)
file_path = 'label_{}.json'.format("2", "iterative_train")
with open(file_path, 'w') as json_file:
    json.dump(data_to_save, json_file)

print(f"Data saved to {file_path}")

# Writing the data to a JSON file.
with open(file_path, 'w') as json_file:
    json.dump(data_to_save, json_file)

print(f"Data saved to {file_path}")
raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
# )

print(train_dataset)

trainer = IterativeRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=reward_config.max_length),
    )
# label_callback = LabelCallback(trainer=trainer)
# trainer.add_callback(label_callback)
trainer.train()
trainer.save_model(reward_config.output_dir + "_epoch1_final_checkpoint")