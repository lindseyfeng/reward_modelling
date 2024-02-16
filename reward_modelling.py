from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import RewardConfig
from reward_trainer import IterativeRewardTrainer
from datasets import load_dataset
from tqdm import tqdm

PAD_TOKEN = '[PAD]'
tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b")
# Assuming `tokenizer` is your tokenizer instance
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



raw_datasets = load_dataset("Anthropic/hh-rlhf")
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

reward_config = RewardConfig(
    output_dir="hh_openllama3b_temp1.2",
    per_device_train_batch_size=4,
    per_device_eval_batch_size = 4,
    # gradient_accumulation_steps = 2, 
    max_length = 512, 
    learning_rate=1e-5,
    report_to="wandb",
    optim="adamw_torch"
)

def preprocess_function(examples):
    new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
            "labels": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, padding=True)
        tokenized_rejected = tokenizer(rejected, padding=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])
        new_examples["labels"].append(1)

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
        eval_dataset=eval_dataset
    )
trainer.custom_train_loop()
trainer.save_model(reward_config.output_dir)