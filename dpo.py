
# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch.nn as nn
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
import json
from trl import DPOTrainer
from dpo_temperature_scaling import _ECELoss, temperature_scale, set_temperature, set_temperature
import wandb
import torch.nn.functional as F

base_dir = "../llama/llama-2-7b"
device = "cuda" if torch.cuda.is_available() else "cpu"

class ECEDP0Trainer(DPOTrainer):
    def __init__(self, *args, beta_update_interval=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_update_interval = beta_update_interval
        self.eval_step_counter = 0
        self.temperature = nn.Parameter((torch.ones(1)*1).to(device))
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Check if it's time to update beta
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if self.eval_step_counter % self.beta_update_interval == 0:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            eval_dataloader = self.data_collator(eval_dataloader.dataset)
            (
                        policy_chosen_logps,
                        policy_rejected_logps,
                        policy_chosen_logits,
                        policy_rejected_logits,
            ) = self.concatenated_forward(self.model, eval_dataloader)
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                eval_dataloader["reference_chosen_logps"],
                eval_dataloader["reference_rejected_logps"],
            )
            print("print", chosen_rewards.tolist(), rejected_rewards.tolist())
            ece = set_temperature(chosen_rewards.tolist(), rejected_rewards.tolist(), self.temperature, script_args.output_dir)
            log_value = self.temperature.detach().cpu().item()
            wandb.log({'temperature_trajectory': self.beta})
            wandb.log({'ece': ece})
        

        # Increment the counter
        self.eval_step_counter += 1
        
        # Now call the original evaluate function
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
    

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps

        ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios
        print("logits in dpo_loss", logits*self.beta)

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards 


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default=base_dir,
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=150, metadata={"help": "the number of warmup steps"})
    optimizer_type: Optional[str] = field(default="rmsprop", metadata={"help": "the optimizer type"})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "num epoch"})
    per_device_train_batch_size: Optional[int] = field(default=1, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=64, metadata={"help": "the lora r parameter"})
    max_target_length: Optional[int] = field(default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"})
    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=2000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=500, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=1, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./dpo_llama7b_results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    train_path: Optional[str] = field(
        default="train_data.json",
    )
    val_path: Optional[str] = field(
        default="val_data.json",
    )

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return Dataset.from_dict(data)




def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: str = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Dahoas/full-hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 10)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }

    return dataset.map(split_prompt_and_responses)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_hh("train", sanity_check=script_args.sanity_check)
    print(train_dataset)

    # 3. Load evaluation dataset
    eval_dataset = get_hh("test", sanity_check=script_args.sanity_check)
    print(eval_dataset)

    # train_dataset = load_json(script_args.train_path)
    # eval_dataset = load_json(script_args.val_path)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        do_eval = True,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        logging_steps=script_args.logging_steps,
        logging_first_step=True,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        save_strategy="epoch",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        remove_unused_columns=False,
        run_name="dpo_llama7b_temp_{}".format(script_args.beta),
        num_train_epochs=script_args.num_train_epochs,
        save_total_limit=1,
        bf16=True
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = ECEDP0Trainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        precompute_ref_log_probs = True, 
    )

    # 6. train
    dpo_trainer.train()

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)