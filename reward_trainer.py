# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import warnings
from dataclasses import FrozenInstanceError, replace
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.optim import AdamW
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalPrediction

from trl.import_utils import is_peft_available
from trl.trainer.reward_config import RewardConfig
from trl.trainer.utils import RewardDataCollatorWithPadding, compute_accuracy
import wandb
from torch.cuda.amp import GradScaler, autocast



wandb.login()
def contains_nan(tensor):
    return torch.isnan(tensor).any().item()



if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

BETA = 0.7
ALPHA = 1e-5
TEMPERATURE = 1/1.2
EPOCH = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
epsilon = 1e-8

class IterativeRewardTrainer(Trainer):
    r"""
    The RewardTrainer can be used to train your custom Reward Model. It is a subclass of the
    `transformers.Trainer` class and inherits all of its attributes and methods. It is recommended to use
    an `AutoModelForSequenceClassification` as the reward model. The reward model should be trained on a dataset
    of paired examples, where each example is a tuple of two sequences. The reward model should be trained to
    predict which example in the pair is more relevant to the task at hand.

    The reward trainer expects a very specific format for the dataset. The dataset should contain two 4 entries at least
    if you don't use the default `RewardDataCollatorWithPadding` data collator. The entries should be named
    - `input_ids_chosen`
    - `attention_mask_chosen`
    - `input_ids_rejected`
    - `attention_mask_rejected`

    Optionally, you can also pass a `margin` entry to the dataset. This entry should contain the margin used to modulate the
    loss of the reward model as outlined in https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/.
    If you don't pass a margin, no margin will be used.
    """

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module]] = None,
        args: Optional[RewardConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        """
        Initialize RewardTrainer.

        Args:
            model (`transformers.PreTrainedModel`):
                The model to train, preferably an `AutoModelForSequenceClassification`.
            args (`RewardConfig`):
                The arguments to use for training.
            data_collator (`transformers.DataCollator`):
                The data collator to use for training. If None is specified, the default data collator (`RewardDataCollatorWithPadding`) will be used
                which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
            train_dataset (`datasets.Dataset`):
                The dataset to use for training.
            eval_dataset (`datasets.Dataset`):
                The dataset to use for evaluation.
            tokenizer (`transformers.PreTrainedTokenizerBase`):
                The tokenizer to use for training. This argument is required if you want to use the default data collator.
            model_init (`Callable[[], transformers.PreTrainedModel]`):
                The model initializer to use for training. If None is specified, the default model initializer will be used.
            compute_metrics (`Callable[[transformers.EvalPrediction], Dict]`, *optional* defaults to `compute_accuracy`):
                The metrics to use for evaluation. If no metrics are specified, the default metric (`compute_accuracy`) will be used.
            callbacks (`List[transformers.TrainerCallback]`):
                The callbacks to use for training.
            optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
                The optimizer and scheduler to use for training.
            preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                The function to use to preprocess the logits before computing the metrics.
            max_length (`int`, defaults to `None`):
                The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
            peft_config (`Dict`, defaults to `None`):
                The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        """
        if type(args) == TrainingArguments:
            warnings.warn(
                "Using `transformers.TrainingArguments` for `args` is deprecated and will be removed in a future version. Please use `RewardConfig` instead.",
                FutureWarning,
            )
            if max_length is not None:
                warnings.warn(
                    "The `max_length` argument is deprecated and will be removed in a future version. Please use the `RewardConfig` to set `max_length` instead.",
                    FutureWarning,
                )
        else:
            if max_length is not None and args.max_length is not None:
                raise ValueError(
                    "You cannot specify both `max_length` and `args.max_length`. Please use the `RewardConfig` to set `max_length` once."
                )
            if max_length is not None and args.max_length is None:
                warnings.warn(
                    "The `max_length` argument is deprecated and will be removed in a future version. Please use the `RewardConfig` to set `max_length` instead.",
                    FutureWarning,
                )
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if not isinstance(model, PeftModel):
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_quantized", False):
                    _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                        inspect.signature(prepare_model_for_kbit_training).parameters
                    )

                    preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                    if not _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        warnings.warn(
                            "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                            "please update to the latest version of peft to use `gradient_checkpointing_kwargs`."
                        )
                    elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                        preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                    model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)

                model = get_peft_model(model, peft_config)

        if compute_metrics is None:
            compute_metrics = compute_accuracy

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default RewardDataCollatorWithPadding"
                )
            if type(args) == TrainingArguments:
                if max_length is None:
                    warnings.warn(
                        "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
                        " It will be set to `512` by default, but you should do it yourself in the future.",
                        UserWarning,
                    )
                    max_length = 512
            else:
                if max_length is None and args.max_length is None:
                    warnings.warn(
                        "When using RewardDataCollatorWithPadding, you should set `max_length` in RewardConfig."
                        " It will be set to `512` by default, but you should do it yourself in the future.",
                        UserWarning,
                    )
                    max_length = 512
                if max_length is None and args.max_length is not None:
                    max_length = args.max_length

            data_collator = RewardDataCollatorWithPadding(tokenizer, max_length=max_length)

            if args.remove_unused_columns:
                try:  # for bc before https://github.com/huggingface/transformers/pull/25435
                    args.remove_unused_columns = False
                except FrozenInstanceError:
                    args = replace(args, remove_unused_columns=False)
                # warn users
                warnings.warn(
                    "When using RewardDataCollatorWithPadding, you should set `remove_unused_columns=False` in your RewardConfig"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                ) 

            self.use_reward_data_collator = True
        else:
            self.use_reward_data_collator = False
        print("optimizers", optimizers)
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        if self.optimizer == None:
            self.optimizer = AdamW(model.parameters(), lr=ALPHA, eps=1e-8)
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_reward_data_collator:
            warnings.warn(
                "The current compute_loss is implemented for RewardDataCollatorWithPadding,"
                " if you are using a custom data collator make sure you know what you are doing or"
                " implement your own compute_loss method."
            )
        # Before forward pass
        if contains_nan(inputs["input_ids_chosen"]) or contains_nan(inputs["attention_mask_chosen"]):
            print("NaN detected in inputs")

        if contains_nan(inputs["input_ids_rejected"]) or contains_nan(inputs["attention_mask_chosen"]):
            print("NaN detected in inputs")

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

        # Compute the softmax probabilities for chosen over rejected items
        exp_logits_chosen = torch.exp(rewards_chosen * TEMPERATURE)
        exp_logits_rejected = torch.exp(rewards_rejected * TEMPERATURE)
        probs_chosen = exp_logits_chosen / (exp_logits_chosen + exp_logits_rejected)
        probs_rejected = exp_logits_rejected / (exp_logits_chosen + exp_logits_rejected)
        print(inputs)
        labels = inputs["label"]  # Assuming labels are provided in inputs
    
        # Compute the loss based on the labels and probabilities
        loss_chosen = -labels * torch.log(probs_chosen)
        loss_rejected = -(1 - labels) * torch.log(probs_rejected)
        loss = (loss_chosen + loss_rejected).mean()
        print("loss", loss)

        # if "margin" in inputs:
        #     loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        # else:
        #     loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, probs_chosen, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss, probs_chosen


    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)
            # After forward pass, before backward

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        # Stack accepted against rejected, mean over logits
        # and softmax to get preferences between accepted and rejected to sum to 1
        logits = torch.stack(logits).mean(dim=2).softmax(dim=0).T

        labels = torch.zeros(logits.shape[0])
        labels = self._prepare_inputs(labels)

        return loss, logits, labels

    def update_labels_with_model_predictions(self, inputs, probs_chosen):
        inputs["label"] = (1-BETA)*inputs["label"] + BETA * probs_chosen


    def append_labels_to_batches(self):
        labels = torch.ones((self._train_batch_size, ), dtype=torch.long).to(device)
        return labels

    from torch.cuda.amp import GradScaler, autocast

    def training_step(self, model, inputs):
        """
        Perform a training step, including updating labels based on model predictions.
        """
        model.train()
        print(inputs)
        inputs = self._prepare_inputs(inputs)
        print(inputs['label'])
        # Compute loss with updated labels

        if 'label' not in inputs:
            inputs['label'] = fetch_labels_for_inputs()
            print("labels not found")
        loss, probs_chosen, outputs = self.compute_loss(model, inputs, return_outputs=True)

        # Update labels within inputs based on model predictions
        self.update_labels_with_model_predictions(inputs, probs_chosen)
        print(inputs['label'])

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # Backward pass
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        wandb.log({"train_loss": loss.item()})

        return loss.detach()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Call the parent class's evaluate method
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Log evaluation results to WandB
        wandb.log({f"{metric_key_prefix}_{k}": v for k, v in eval_result.items()})
        
        return eval_result

    

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        return dataset
        # if not self.args.remove_unused_columns:
        #     return dataset
        # self._set_signature_columns_if_needed()
        # signature_columns = self._signature_columns
        # signature_columns.add("label")

        # ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        # if len(ignored_columns) > 0:
        #     dset_description = "" if description is None else f"in the {description} set"
        #     logger.info(
        #         f"The following columns {dset_description} don't have a corresponding argument in "
        #         f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
        #         f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
        #         " you can safely ignore this message."
        #     )

        # columns = [k for k in signature_columns if k in dataset.column_names]

        # if version.parse(datasets.__version__) < version.parse("1.4.0"):
        #     dataset.set_format(
        #         type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
        #     )
        #     return dataset
        # else:
        #     return dataset.remove_columns(ignored_columns)

        

    