from dataclasses import dataclass
from typing import Any
import torch
from src.utils import *

@dataclass
class DataCollator:
    tokenizer:Any
    clip_processor:Any
    max_prompt_len:int
    max_answer_len:int
    
    def __call__(self,batch):
        images=[ensure_pil_rgb(x["image"]) for x in batch]
        clip_inputs=self.clip_processor(images=images,return_tensors="pt")
        pixel_values=clip_inputs["pixel_values"]

        prompts=[x["prompt"] for x in batch]
        answers=[x["answer"] for x in batch]

        prompt_toks=self.tokenizer(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_len,
            # return_tensors="pt"
        )

        answer_toks=self.tokenizer(
            answers,
            padding=False,
            truncation=True,
            max_length=self.max_answer_len,
            # return_tensors="pt"
        )

        input_ids=[]
        labels=[]

        for prompt_id ,answer_id in zip(prompt_toks["input_ids"],
                                        answer_toks["input_ids"]):
            ids=torch.tensor(prompt_id+answer_id,dtype=torch.long)
            label=torch.tensor([-100]*len(prompt_id)+answer_id,dtype=torch.long)
            input_ids.append(ids)
            labels.append(label)

        pad_id=self.tokenizer.pad_token_id
        input_ids=torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=pad_id
        )
        labels_pad=torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100
        )
        attention_mask=(input_ids!=pad_id).long()

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_pad,
        }
