from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import json
from src.utils import ensure_pil_rgb, extract_json, find_subsequence
@dataclass
class DataCollator:
    tokenizer: Any
    image_processor: Any
    max_prompt_len: int
    max_answer_len: int

    disease_weight=8.0
    evidence_weight=3.0
    base_answer_weight=1.0

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

        weights_list = []
        disease_id_list = []

        for ans_text,p_ids, a_ids in zip(answers,prompt_only["input_ids"], answer_only["input_ids"]):
            w=torch.zeros(len(p_ids)+len(a_ids),dtype=torch.float32)
            w[len(p_ids):]=self.base_answer_weight

            disease_id=-1
            obj=extract_json(ans_text)
            if isinstance(obj,dict):
                d=obj.get('disease',None)
                if isinstance(d,str):
                    dl=d.strip().lower()
                    if "healthy" in dl:
                        disease_id = 0
                    elif "downy" in dl:
                        disease_id = 1
                    elif "powdery" in dl:
                        disease_id = 2
                    else:
                        disease_id = 3  # unknown
                try:
                    if isinstance(d,str) and len(a_ids)>0:
                        d_tok=self.tokenizer(d,add_special_tokens=False)["input_ids"]
                        hits=find_subsequence(a_ids,d_tok)
                        if hits:
                            s=hits[0]
                            e=s+len(d_tok)
                            w[len(p_ids)+s:len(p_ids)+e]=self.disease_weight
                    
                    ev=obj.get("evidence",None)
                    if ev is not None and len(a_ids)>0:
                        ev_text=json.dumps(ev,ensure_ascii=False)
                        ev_tok=self.tokenizer(ev_text,add_special_tokens=False)["input_ids"]
                        hits=find_subsequence(a_ids,ev_tok)
                        if hits:
                            s=hits[0]
                            e=s+len(ev_tok)
                            w[len(p_ids)+s:len(p_ids)+e]=self.evidence_weight
                except Exception:
                    pass

            ids = torch.tensor(p_ids + a_ids, dtype=torch.long)

            lab = torch.tensor([-100] * len(p_ids) + a_ids, dtype=torch.long)

            input_ids_list.append(ids)
            labels_list.append(lab)
            prompt_ids_list.append(torch.tensor(p_ids, dtype=torch.long))
            weights_list.append(w)
            disease_id_list.append(disease_id)

        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad_id).long()

        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(prompt_ids_list, batch_first=True, padding_value=pad_id)
        prompt_attention_mask = (prompt_input_ids != pad_id).long()

        loss_weights=torch.nn.utils.rnn.pad_sequence(weights_list,batch_first=True,padding_value=0.0)
        disease_id=torch.tensor(disease_id_list,dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "loss_weights":loss_weights,
            "disease_id":disease_id,
        }
