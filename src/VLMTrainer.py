from transformers import Trainer
import torch

class VLMTrainer(Trainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        outputs=model(**inputs)
        loss=outputs.loss

        labels=inputs["labels"]

        loss_tokens=(labels!=-100).sum()
        total_tokens=labels.numel()
        self.log({
            "loss_tokens": loss_tokens.item(),
            "ratio": (loss_tokens/total_tokens).item(),
        })
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
        labels=inputs.get("labels",None)

        input_ids=inputs["input_ids"]
        attention_mask=inputs["attention_mask"]
        pixel_values=inputs["pixel_values"]

        DEVICE=input_ids.device
        model.eval()

        with torch.no_grad():
            clip_emb=self.clip.get_image_features(pixel_values=pixel_values)
            clip_emb=clip_emb/(clip_emb.norm(dim=-1,keepdim=True)+1e-6)

        prefix_embeds=model.mapping_net(clip_emb)

        embed_layer=model.qwen.get_input_embeddings()
        token_embeds=embed_layer(input_ids)

        inputs_embeds=torch.cat([prefix_embeds,token_embeds],dim=1)

        B=input_ids.size(0)

        prefix_attention=torch.ones((B,self.prefix_len),dtype=attention_mask.dtype,device=DEVICE)
        attention_mask=torch.cat([prefix_attention,attention_mask],dim=1)

        prefix_labels=torch.full((B,self.prefix_len),-100,dtype=labels.dtype,device=DEVICE)
        labels=torch.cat([prefix_labels,labels],dim=1)

        genrated_ids=model.qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            do_sample=False,
            use_cache=True,
        )
        return None,genrated_ids,labels
