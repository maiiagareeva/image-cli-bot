import os
import json
from dataclasses import dataclass
from typing import Any,Dict,List,Optional
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments,Trainer

from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from transformers import CLIPModel,CLIPProcessor
from huggingface_hub import hf_hub_download
from datasets import load_dataset

BASE_MODEL="Qwen/Qwen3-1.7B"
# DATASET="qingwuuu/ngld-grape-leaf-vlm"
# DATASET="qingwuuu/ngld-grape-leaf-vlm-w-img"
DATASET="qingwuuu/ngld-grape-leaf-vlm-w-img-without-diff-ref"
OUT_DIR="./results_mm_1"
NEW_MODEL="./qwen3-1.7B-ngld-lora-1"

CLIP_NAME="openai/clip-vit-base-patch32"
PREFIX_LEN=20
# QWEN_DIM=2048
MAX_PROMPT_LEN = 256
MAX_ANSWER_LEN = 256

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def ensure_pil_rgb(image):
    if isinstance(image,Image.Image):
        return image.convert("RGB")
    if isinstance(image,dict) and "bytes" in image and image["bytes"] is not None:
        from io import BytesIO
        return Image.open(BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image,dict) and "path" in image and image["path"] is not None:
        return Image.open(image["path"]).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)}")

class MappingNet(nn.Module):
    def __init__(self,d_clip,hidden_size,p):
        super().__init__()
        self.p=p
        self.hidden_size=hidden_size
        self.net=nn.Sequential(
            nn.Linear(d_clip,4*hidden_size),
            nn.GELU(),
            nn.Linear(4*hidden_size,p*hidden_size),
        )
    def forward(self,clip_emb):
        B=clip_emb.shape[0]
        x=self.net(clip_emb)
        x=x.view(B,self.p,self.hidden_size)
        return x
    
@dataclass
class datacollator:
    tokenizer:Any
    clip_processor:Any
    max_prompt_len:int=MAX_PROMPT_LEN
    max_answer_len:int=MAX_ANSWER_LEN
    
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

class QwenwithPrefix(nn.Module):
    def __init__(self,qwen,clip_model,mapping_net,prefix_len=PREFIX_LEN):
        super().__init__()
        self.qwen=qwen
        self.clip=clip_model
        self.mapping_net=mapping_net
        self.prefix_len=prefix_len

        for p in self.clip.parameters():
            p.requires_grad=False
        self.clip.eval()

    def forward(self,input_ids,attention_mask,labels,pixel_values,**kwargs):
        DEVICE=input_ids.device

        with torch.no_grad():
            clip_emb=self.clip.get_image_features(pixel_values=pixel_values)
            clip_emb=clip_emb/(clip_emb.norm(dim=-1,keepdim=True)+1e-6)

        prefix_embeds=self.mapping_net(clip_emb)

        embed_layer=self.qwen.get_input_embeddings()
        token_embeds=embed_layer(input_ids)

        inputs_embeds=torch.cat([prefix_embeds,token_embeds],dim=1)

        B=input_ids.size(0)

        prefix_attention=torch.ones((B,self.prefix_len),dtype=attention_mask.dtype,device=DEVICE)
        attention_mask=torch.cat([prefix_attention,attention_mask],dim=1)

        prefix_labels=torch.full((B,self.prefix_len),-100,dtype=labels.dtype,device=DEVICE)
        labels=torch.cat([prefix_labels,labels],dim=1)

        #test
        loss_tokens=(labels!=-100).sum()
        total_tokens=labels.numel()
        print("loss_tokens: ",loss_tokens.item(),
              "ratio: ",(loss_tokens/total_tokens).item()
              )

        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

def set_requires_grad(module,flag):
    for p in module.parameters():
        p.requires_grad=flag

def main():
    ds = load_dataset(DATASET)
    train_ds = ds["train"]
    eval_ds=ds["val"]

    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    qwen=AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization_config,
        device_map={"":0},
        trust_remote_code=True,
    )
    qwen.config.use_cache=False
    qwen=prepare_model_for_kbit_training(qwen)

    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="right"

    peft_config=LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","v_proj"]
    )
    qwen=get_peft_model(qwen,peft_config)

    clip_model=CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    clip_processor=CLIPProcessor.from_pretrained(CLIP_NAME)
    for p in clip_model.parameters():
        p.requires_grad=False


    d_clip=clip_model.config.projection_dim
    d_qwen=qwen.config.hidden_size
    mapping_net=MappingNet(d_clip=d_clip,hidden_size=d_qwen,p=PREFIX_LEN).to(device)

    STAGE="MAPPING_TRAIN"

    if STAGE=="MAPPING_TRAIN":
        set_requires_grad(qwen,False)
        set_requires_grad(mapping_net,True)
    else:
        set_requires_grad(mapping_net,False)
        mapping_net.load_state_dict(torch.load("/users/4/shen0574/lora_demo/qwen3-1.7B-ngld-lora/mapping.pt",map_location=device))
    
    trainable=sum(p.numel() for p in qwen.parameters() if p.requires_grad)
    total=sum(p.numel() for p in qwen.parameters())
    
    qwen_embed_model=qwen.base_model.model if hasattr(qwen,"base_model") else qwen

    model = QwenwithPrefix(qwen=qwen, 
                           clip_model=clip_model, 
                           mapping_net=mapping_net, 
                           prefix_len=PREFIX_LEN
                           )

    collator=datacollator(
        tokenizer=tokenizer,
        clip_processor=clip_processor,
        max_prompt_len=MAX_PROMPT_LEN,
        max_answer_len=MAX_ANSWER_LEN,
    )

    args=TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=25,
        save_steps=200,
        # evaluation_strategy = "steps" if eval_ds is not None else "no",
        eval_steps=200 if eval_ds is not None else None,
        fp16=True,
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=True,

        save_safetensors=False,
    )

    trainer=Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    #test
    before = mapping_net.net[0].weight.detach().clone()
    #train
    trainer.train()
    #test
    after = mapping_net.net[0].weight.detach()
    print(torch.norm(after - before))

    model.qwen.save_pretrained(NEW_MODEL)
    tokenizer.save_pretrained(NEW_MODEL)
    torch.save(mapping_net.state_dict(),os.path.join(NEW_MODEL,"mapping.pt"))
    print("save to: ",NEW_MODEL)

if __name__=="__main__":
    main()
