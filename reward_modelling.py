from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import RewardConfig
from reward_trainer import IterativeRewardTrainer, labelCallback
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

BETA = 0.7
ALPHA = 1e-5
TEMPERATURE = 1
EPOCH = 4

PAD_TOKEN = '[PAD]'
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b")
# Assuming `tokenizer` is your tokenizer instance
if tokenizer.pad_token is None:
    tokenizer.pad_token = PAD_TOKEN


raw_datasets = load_dataset("Anthropic/hh-rlhf")
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

reward_config = RewardConfig(

    output_dir="hh_openllama3b_temp1.2",
    per_device_train_batch_size= 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 4, 
    max_length = 512, 
    learning_rate=1e-5,
    report_to="wandb",
    optim="adamw_torch"
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
        print("hello?its me")
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
                print("yes2")
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
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, padding = "max_length", max_length = reward_config.max_length)
        tokenized_rejected = tokenizer(rejected, padding = "max_length", max_length = reward_config.max_length)
        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["label"].append(torch.tensor([[1]]))

    return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
train_dataset = raw_datasets["train"].shuffle(seed=42).select(range(40000)) ####validate code is fine
eval_dataset = raw_datasets["test"]

# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
# )

print(train_dataset)

wandb.init(project='BETA_{}_ALPHA_{}_TEMP_{}_EPOCH'.format(BETA, ALPHA, TEMPERATURE, EPOCH), config={
            "learning_rate": ALPHA,
            "epochs": EPOCH,
            "batch_size": reward_config.per_device_train_batch_size,
            # Add other hyperparameters here,
        })

trainer = IterativeRewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[labelCallback()],
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer, max_length=reward_config.max_length),

    )

trainer.train()
trainer.save_model(reward_config.output_dir)