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
from trl import RewardTrainer
from trl.import_utils import is_peft_available
from trl.trainer.reward_config import RewardConfig
from trl.trainer.utils import RewardDataCollatorWithPadding, compute_accuracy
from torch.cuda.amp import GradScaler, autocast



def contains_nan(tensor):
    return torch.isnan(tensor).any().item()


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

BETA = 0.7
ALPHA = 1e-5
EPOCH = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
epsilon = 1e-8

class IterativeRewardTrainer(RewardTrainer):
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
        exp_logits_chosen = torch.exp(rewards_chosen)
        exp_logits_rejected = torch.exp(rewards_rejected)
        probs_chosen = exp_logits_chosen / (exp_logits_chosen + exp_logits_rejected)
        probs_rejected = exp_logits_rejected / (exp_logits_chosen + exp_logits_rejected)
        labels = inputs["label"]  # Assuming labels are provided in inputs
        labels = labels.unsqueeze(-1)
        # Compute the loss based on the labels and probabilities
        loss_chosen = -labels * torch.log(probs_chosen)
        loss_rejected = -(1 - labels) * torch.log(probs_rejected)
        loss = (loss_chosen + loss_rejected).mean()

        # if "margin" in inputs:
        #     loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected - inputs["margin"]).mean()
        # else:
        #     loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()

        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_chosen,
                "rewards_rejected": rewards_rejected,
            }
        return loss


    def update_labels(self, inputs, model):
        print("labels are being updated")
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
        exp_logits_chosen = torch.exp(rewards_chosen)
        exp_logits_rejected = torch.exp(rewards_rejected)
        probs_chosen = exp_logits_chosen / (exp_logits_chosen + exp_logits_rejected)
        inputs["label"] = (1-BETA)*inputs["label"] + BETA * probs_chosen

    def _invoke_callbacks(self):
        args = self.args
        state = self.state
        control = self.control

        # Now, ensure that when callbacks are invoked, 'trainer=self' is included
        for callback in self.callback_handler.callbacks:
            print("hello?")
            callback.on_epoch_end(args, state, control, trainer=self)

    
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

class labelCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print("hi!")
        if 'trainer' in kwargs:
            trainer = kwargs['trainer']
            train_dataloader = trainer.get_train_dataloader()
            for batch in train_dataloader:
                inputs = trainer._prepare_inputs(batch)
                print("label before: ", inputs["label"])
                trainer.update_labels(inputs, trainer.model)
                print("label after: ", inputs["label"])



    