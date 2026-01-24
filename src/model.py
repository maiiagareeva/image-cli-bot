import torch
import torch.nn as nn
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from transformers import CLIPModel,CLIPProcessor
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments,Trainer
from src.utils import *
from QFormer.qformer import QFormer

class QwenwithPrefix(nn.Module):
    def __init__(self,qwen,clip_model,qformer,projector,prefix_len):
        super().__init__()
        self.qwen=qwen
        self.clip=clip_model
        self.qformer=qformer
        self.projector=projector
        self.prefix_len=prefix_len

        for p in self.clip.parameters():
            p.requires_grad=False
        self.clip.eval()

    def forward(self,input_ids,attention_mask,labels,pixel_values,**kwargs):
        DEVICE=input_ids.device

        with torch.no_grad():
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state[:,1:,:]
        
            B, N, _ = image_embeds.shape
            image_atts = torch.ones((B, N), dtype=torch.long, device=image_embeds.device)

        query_embeds =self.qformer(image_embeds,image_atts)
        prefix_embes=self.projector(query_embeds)

        embed_layer=self.qwen.get_input_embeddings()
        token_embeds=embed_layer(input_ids)

        inputs_embeds=torch.cat([prefix_embes ,token_embeds],dim=1)

        B=input_ids.size(0)

        prefix_attention=torch.ones((B,self.prefix_len),dtype=attention_mask.dtype,device=DEVICE)
        attention_mask=torch.cat([prefix_attention,attention_mask],dim=1)

        prefix_labels=torch.full((B,self.prefix_len),-100,dtype=labels.dtype,device=DEVICE)
        labels=torch.cat([prefix_labels,labels],dim=1)

        return self.qwen(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

def build_model(global_,model,data,training,stage,device):
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    qwen=AutoModelForCausalLM.from_pretrained(
        model.base_model,
        quantization_config=quantization_config,
        device_map={"":0},
        trust_remote_code=True,
    )
    qwen.config.use_cache=False
    qwen=prepare_model_for_kbit_training(qwen)

    lora_config=model.lora
    peft_config=LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        target_modules=lora_config.target_modules
    )
    qwen=get_peft_model(qwen,peft_config)

    clip_model=CLIPModel.from_pretrained(model.clip_model).to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad=False

    d_vision=clip_model.vision_model.config.hidden_size
    d_qwen=qwen.config.hidden_size
    qformer=QFormer(
        num_query_tokens=model.prefix_len,
        vision_hidden_dim=d_vision,
        qformer_hidden_dim=768,
        num_hidden_layers=12,
        cross_attention_freq= 2,
        pretrained_bert= "bert-base-uncased"
        ).to(device)
    
    projector=nn.Linear(768,d_qwen).to(device)

    if stage.name=="QUERY_TRAIN":
        set_requires_grad(qwen,False)
        set_requires_grad(qformer,True)
        set_requires_grad(projector,True)
    else:
        set_requires_grad(qformer,False)
        qformer.load_state_dict(torch.load(stage.qformer_ckpt,map_location=device))

    model=QwenwithPrefix(qwen=qwen,
                         clip_model=clip_model,
                         qformer=qformer,
                         projector=projector,
                         prefix_len=model.prefix_len
                        )
    
    return model,qformer
