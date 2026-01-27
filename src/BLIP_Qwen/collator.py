from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from src.utils import ensure_pil_rgb

@dataclass
class DataCollator:
    tokenizer: Any
    image_processor: Any
    max_prompt_len: int
    max_answer_len: int

    def __call__(self, batch):
        images = [ensure_pil_rgb(x["image"]) for x in batch]
        pixel_values = self.image_processor(images=images, return_tensors="pt").pixel_values

        prompts = [x["prompt"] for x in batch]
        answers = [x["answer"] for x in batch]

        prompt_only = self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_len,
            add_special_tokens=True,
        )

        answer_only = self.tokenizer(
            answers,
            padding=False,
            truncation=True,
            max_length=self.max_answer_len,
            add_special_tokens=False,
        )

        input_ids_list = []
        labels_list = []
        prompt_ids_list = []

        for p_ids, a_ids in zip(prompt_only["input_ids"], answer_only["input_ids"]):
            ids = torch.tensor(p_ids + a_ids, dtype=torch.long)

            lab = torch.tensor([-100] * len(p_ids) + a_ids, dtype=torch.long)

            input_ids_list.append(ids)
            labels_list.append(lab)
            prompt_ids_list.append(torch.tensor(p_ids, dtype=torch.long))

        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad_id).long()

        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(prompt_ids_list, batch_first=True, padding_value=pad_id)
        prompt_attention_mask = (prompt_input_ids != pad_id).long()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }
