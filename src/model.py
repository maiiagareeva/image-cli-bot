import torch
import torch.nn as nn
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from transformers import CLIPModel,CLIPProcessor
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig,TrainingArguments,Trainer
from src.utils import *

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
    

class QwenwithPrefix(nn.Module):
    def __init__(self,qwen,clip_model,mapping_net,prefix_len):
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

def build_model(global_,model,data,training,stage,device):
    #Qwen 4bit
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

    #Lora
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

    #CLIP
    clip_model=CLIPModel.from_pretrained(model.clip_model).to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad=False

    #mapping net
    d_clip=clip_model.config.projection_dim
    d_qwen=qwen.config.hidden_size
    mapping_net=MappingNet(d_clip=d_clip,hidden_size=d_qwen,p=model.prefix_len).to(device)

    #training stage
    if stage.name=="MAPPING_TRAIN":
        set_requires_grad(qwen,False)
        set_requires_grad(mapping_net,True)
    else:
        set_requires_grad(mapping_net,False)
        mapping_net.load_state_dict(torch.load(stage.mapping_cpkt,map_location=device))

    model=QwenwithPrefix(qwen=qwen,
                         clip_model=clip_model,
                         mapping_net=mapping_net,
                         prefix_len=model.prefix_len
                        )
    
    return model,mapping_net
    