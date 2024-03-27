import torch
from torch import optim, nn
from transformers import AutoTokenizer, AutoModelForCausalLM, PhiForSequenceClassification, HfArgumentParser, TrainingArguments
from datasets import load_dataset, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import os
from dataclasses import dataclass, field
from typing import Dict, Optional
import json
from trl import DPOTrainer
import wandb
from peft import LoraConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

from torch.utils.data.dataloader import default_collate

def logits_to_list(logits_tensor):
    logits_list = logits_tensor.detach().cpu().tolist()
    # Flatten the list since the original tensor has a shape of [10, 1]
    return logits_list

# Define and parse arguments.
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
     # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=5, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

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
        dataset = dataset.select(range(min(len(dataset), 100)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        return {
            "prompt": sample["prompt"],
            "chosen": sample["chosen"],
            "rejected": sample["rejected"],
        }

    return dataset.map(split_prompt_and_responses)


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    """  
    def __init__(self, n_bins=5):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0.5, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1].to(device)
        self.bin_uppers = bin_boundaries[1:].to(device)

    def forward(self, logits, labels):
        softmaxes = torch.sigmoid(logits)
        print(softmaxes)
        confidences, _ = torch.max(softmaxes, 1)
        print(confidences)
        predictions = (softmaxes >= 0.5001).long() 
        print(predictions.shape)
        print(predictions)
        print(labels)

        accuracies = torch.argmax(predictions, dim=1).eq(labels)


        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        percentage_true = torch.mean(accuracies.float()) * 100  
        print("accuracy : ", percentage_true.item())
        return ece


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


def temperature_scale(logits, temperature):
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1)).cuda()
    return logits / temperature

def set_temperature_trl(inputs, model, temperature):
    beta = 0.1
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()
    model.eval()
    with torch.no_grad():
        logits_list = []
        labels_list = []
        print(inputs)
        concatenated_input_ids = torch.cat([inputs["chosen_input_ids"], inputs["rejected_input_ids"]], dim=0)
        print(concatenated_input_ids)
        concatenated_attention_mask = torch.cat([inputs["chosen_attention_mask"], inputs["rejected_attention_mask"]], dim=0)
        print(concatenated_attention_mask)
            
            # Assuming you're not using an encoder-decoder model or additional inputs like 'decoder_input_ids'.
            # If you do, handle them similarly to 'input_ids' and 'attention_mask'.

            # Infer model once with concatenated inputs
        concatenated_logits = model(input_ids=concatenated_input_ids, attention_mask=concatenated_attention_mask, return_dict=True).logits

            # Split logits back into 'chosen' and 'rejected' parts
        len_chosen = inputs["chosen_input_ids"].shape[0]  # Number of chosen inputs
        concatenated_input_ids = torch.cat([inputs["chosen_input_ids"], inputs["rejected_input_ids"]], dim=0)
        concatenated_attention_mask = torch.cat([inputs["chosen_attention_mask"], inputs["rejected_attention_mask"]], dim=0)
            
            # Assuming you're not using an encoder-decoder model or additional inputs like 'decoder_input_ids'.
            # If you do, handle them similarly to 'input_ids' and 'attention_mask'.

            # Infer model once with concatenated inputs
        concatenated_logits = model(input_ids=concatenated_input_ids, attention_mask=concatenated_attention_mask, return_dict=True).logits

            # Split logits back into 'chosen' and 'rejected' parts
        len_chosen = inputs["chosen_input_ids"].shape[0]  # Number of chosen inputs
        rewards_chosen = concatenated_logits[:len_chosen]
        rewards_rejected = concatenated_logits[len_chosen:]
        chosen_logprob = get_logps(rewards_chosen, inputs["chosen_labels"])
        reject_logprob = get_logps(rewards_rejected, inputs["rejected_labels"])
        ref_chosen_logprob = inputs["reference_chosen_logps"]
        ref_reject_logprob = inputs["reference_rejected_logps"]
        pos_logits = ((chosen_logprob-ref_chosen_logprob)-(reject_logprob-ref_reject_logprob))*beta
        neg_logits = -pos_logits
        logits_list.append(torch.cat((pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)), dim=-1))
            # Convert logits list to tensor and labels list to tensor
        logits = torch.cat(logits_list, dim=0).squeeze(1).to(device)  # This is your tensor from logits_list
        print(logits)

        N, _ = logits.shape
        labels_list += [0] * N # Assuming binary labels, adjust as necessary
        labels = torch.tensor(labels_list).cuda()


        # print(labels)
            # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f ECE: %.3f' %  (before_temperature_nll, before_temperature_ece))

            # Optimize the temperature
        print(temperature.is_leaf) 
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(temperature_scale(logits, temperature), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

            # Calculate NLL after temperature scaling
        after_temperature_nll = nll_criterion(temperature_scale(logits, temperature), labels).item()
        after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
        print('Optimal temperature: %.3f' % temperature.item())
        print('After temperature - NLL: %.3f ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        return before_temperature_ece

def set_temperature(valid_loader, model, temperature, pretrained_model_name_or_path):
    beta = 0.1
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = _ECELoss().cuda()
    with torch.no_grad():
        logits_list = []
        labels_list = []
        logits_to_save = []
        for inputs in valid_loader:
            input_ids_chosen_tensor = inputs["chosen_input_ids"]
            attention_mask_chosen_tensor = inputs["chosen_attention_mask"]
            input_ids_rejected_tensor = inputs["rejected_input_ids"]
            attention_mask_rejected_tensor = inputs["rejected_attention_mask"]
            prompt_tensor = inputs["rejected_input_ids"]
            chosen_label = inputs["chosen_labels"]
            reject_label = inputs["rejected_labels"]
            rewards_chosen = model(input_ids=input_ids_chosen_tensor, attention_mask=attention_mask_chosen_tensor, return_dict=True).logits
            rewards_rejected = model(input_ids=input_ids_rejected_tensor, attention_mask=attention_mask_rejected_tensor, return_dict=True).logits
            chosen_logprob = get_logps(rewards_chosen, chosen_label)
            reject_logprob = get_logps(rewards_rejected, reject_label)
            ref_chosen_logprob = inputs["reference_chosen_logps"]
            ref_reject_logprob = inputs["reference_rejected_logps"]
            pos_logits = ((chosen_logprob-ref_chosen_logprob)-(reject_logprob-ref_reject_logprob))*beta
            logits_to_save.extend(logits_to_list(pos_logits))
            neg_logits = -pos_logits
            logits_list.append(torch.cat((pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)), dim=-1))
            # Convert logits list to tensor and labels list to tensor
        # llama3b
        logits = torch.cat(logits_list, dim=0).squeeze(1)  # This is your tensor from logits_list
        print(logits)

        N, _ = logits.shape
        labels_list += [0] * N # Assuming binary labels, adjust as necessary
        labels = torch.tensor(labels_list).cuda()


        # print(labels)
            # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print('Before temperature - NLL: %.3f ECE: %.3f' %  (before_temperature_nll, before_temperature_ece))

            # Optimize the temperature
        print(temperature.is_leaf) 
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(temperature_scale(logits, temperature), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

            # Calculate NLL after temperature scaling
        after_temperature_nll = nll_criterion(temperature_scale(logits, temperature), labels).item()
        after_temperature_ece = ece_criterion(temperature_scale(logits, temperature), labels).item()
        print('Optimal temperature: %.3f' % temperature.item())
        print('After temperature - NLL: %.3f ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        data_to_save = {
            "logits": logits_to_save,
            "ece": before_temperature_ece,
            "optimal_temp": temperature.item()
        }
        file_path = 'logits_scores_{}_{}.json'.format(pretrained_model_name_or_path.replace("/", "_"), "test")

            # Writing the data to a JSON file.
        with open(file_path, 'w') as json_file:
            json.dump(data_to_save, json_file)
        return before_temperature_ece

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


class Temperature_scaling_DP0Trainer(DPOTrainer):
    def __init__(self, *args, beta_update_interval=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = nn.Parameter((torch.ones(1)).to(device))
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Check if it's time to update beta
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        ece = set_temperature(eval_dataloader, self.model, self.temperature, script_args.model_file)
        log_value = self.temperature.detach().cpu().item()      
        # Now call the original evaluate function
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)



if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    ref_file = script_args.ref_file
    model_file= script_args.model_file
        # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_file,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_file,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(ref_file)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load evaluation dataset
    eval_dataset = get_hh("test", sanity_check=script_args.sanity_check)
    print(eval_dataset)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        do_eval = True,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir="./script_args.output_dir",
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        remove_unused_columns=False,
        run_name="dpo_iterative_llama7b_temperature_scaling",
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
    dpo_trainer = Temperature_scaling_DP0Trainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        precompute_ref_log_probs = True, 
    )

    # 6. train
    dpo_trainer.evaluate()

