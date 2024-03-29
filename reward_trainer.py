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

# class LabelCallback(TrainerCallback):
#     def __init__(self, trainer=None):
#         super().__init__()
#         self.trainer = trainer

#     def update_dataset_with_new_labels(dataset, new_labels):
#         # This creates a new dataset with updated labels
#         updated_dataset = dataset.map(lambda examples, indices: {"label": new_labels[indices]}, with_indices=True)
#         return updated_dataset

#     def update_labels(self, model, dataset):
#         print("labels are being updated")
#         device = model.device

#         # Placeholder for new labels
#         new_labels = []

#         for inputs in dataset:
#             # Prepare inputs for model prediction
#             input_ids_chosen = torch.tensor([inputs["input_ids_chosen"]]).to(device)
#             attention_mask_chosen = torch.tensor([inputs["attention_mask_chosen"]]).to(device)
#             input_ids_rejected = torch.tensor([inputs["input_ids_rejected"]]).to(device)
#             attention_mask_rejected = torch.tensor([inputs["attention_mask_rejected"]]).to(device)

#             with torch.no_grad():  # Ensure no gradients are computed
#                 rewards_chosen = model(input_ids=input_ids_chosen, attention_mask=attention_mask_chosen, return_dict=True)["logits"]
#                 rewards_rejected = model(input_ids=input_ids_rejected, attention_mask=attention_mask_rejected, return_dict=True)["logits"]

#             # Compute softmax probabilities for chosen over rejected items
#             exp_logits_chosen = torch.exp(rewards_chosen)
#             exp_logits_rejected = torch.exp(rewards_rejected)
#             probs_chosen = exp_logits_chosen / (exp_logits_chosen + exp_logits_rejected)

#             # Calculate the updated label based on some logic; you might need to adjust this
#             updated_label = (1 - BETA) * inputs["label"] + BETA * probs_chosen.squeeze().cpu().numpy()
#             new_labels.append(updated_label)

#         # Update the dataset with new labels
#         def update_labels(example, idx):
#             example["label"] = new_labels[idx]
#             return example
#         updated_dataset = dataset.map(update_labels, with_indices=True)
#         return updated_dataset

    

#     def on_epoch_end(self, args, state, control, **kwargs):
#         print("hi!")
#         if self.trainer:
#             print("yo!")
#             new_labels = self.update_labels(self.trainer.model, self.trainer.dataset)
    
#             # Update the dataset with the new labels
#             dataset = self.update_dataset_with_new_labels(self.trainer.dataset, new_labels)

#             # Create a new DataLoader with the updated dataset for the next epoch
#             dataloader = DataLoader(dataset, batch_size=self.trainer.args.per_device_train_batch_size, shuffle=True)
    