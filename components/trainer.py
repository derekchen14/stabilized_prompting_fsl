#!/usr/bin/env python3
#
import os, sys, pdb
import numpy as np
import random
import math

from torch import nn, no_grad
from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger
from components.detector import ExemplarDetective

from utils.help import *
from utils.process import process_data, get_dataloader
from utils.arguments import solicit_params
from utils.load import load_tokenizer, load_model, load_data, load_best_model, load_support
from utils.evaluate import eval_quantify, eval_qualify, test_quantify, parse_output
from assets.static_vars import debug_break
from transformers.deepspeed import deepspeed_init, deepspeed_reinit, is_deepspeed_zero3_enabled
from transformers import Trainer



class DSTrainer(Trainer):
  # def get_train_dataloader(self) -> DataLoader:
  #   train_dataset = self.train_dataset
  #   train_sampler = self._get_train_sampler()
  #   collate = train_dataset.collate_func
  #   dataloader = DataLoader(dataset, sampler=sampler, 
  #                           batch_size=args.batch_size, collate_fn=collate,
  #                           num_workers=self.args.dataloader_num_workers,)
  #   print(f"Loaded {split} data with {len(dataloader)} batches")
  #   return DataLoader(
  #           train_dataset,
  #           batch_size=self.args.train_batch_size,
  #           sampler=train_sampler,
  #           collate_fn=collate,
  #           drop_last=self.args.dataloader_drop_last,
  #           num_workers=self.args.dataloader_num_workers,
  #           pin_memory=self.args.dataloader_pin_memory,
  #       )

  # def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
  #     """
  #     Returns the evaluation [`~torch.utils.data.DataLoader`].

  #     Subclass and override this method if you want to inject some custom behavior.

  #     Args:
  #         eval_dataset (`torch.utils.data.Dataset`, *optional*):
  #             If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
  #             the `model.forward()` method are automatically removed. It must implement `__len__`.
  #     """
  #     if eval_dataset is None and self.eval_dataset is None:
  #         raise ValueError("Trainer: evaluation requires an eval_dataset.")
  #     eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

  #     if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
  #         eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

  #     collate = train_dataset.collate_func

  #     eval_sampler = self._get_eval_sampler(eval_dataset)

  #     return DataLoader(
  #         eval_dataset,
  #         sampler=eval_sampler,
  #         batch_size=self.args.eval_batch_size,
  #         collate_fn=self.data_collator,
  #         drop_last=self.args.dataloader_drop_last,
  #         num_workers=self.args.dataloader_num_workers,
  #         pin_memory=self.args.dataloader_pin_memory,
  #     )


  def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    labels = inputs.get("labels")
    # forward pass
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    return (loss, outputs) if return_outputs else loss