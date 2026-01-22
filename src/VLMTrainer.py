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
        model.eval()

        if prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys,
            )
        
        labels=inputs.get("labels",None)

        input_ids=inputs["input_ids"]
        attention_mask=inputs["attention_mask"]
        pixel_values=inputs["pixel_values"]

        DEVICE=input_ids.device

        with torch.no_grad():
            vision_outputs=model.clip.vision_model(pixel_values=pixel_values)
            image_embeds=vision_outputs.last_hidden_state[:,1:,:]

            B,N,_=image_embeds.shape
            image_atts=torch.ones((B,N),dtype=torch.long,device=image_embeds.device)

            query_embeds=model.qformer(image_embeds,image_atts)
            prefix_embeds=model.projector(query_embeds)

            embed_layer=model.qwen.get_input_embeddings()
            token_embeds=embed_layer(input_ids)

            inputs_embeds=torch.cat([prefix_embeds,token_embeds],dim=1)

            B,P,H=prefix_embeds.shape
            T_pad=input_ids.shape[1]
            T=attention_mask.sum(dim=1).max().item()

            prefix_attention=torch.ones((B,P),dtype=attention_mask.dtype,device=DEVICE)
            attention_mask=torch.cat([prefix_attention,attention_mask],dim=1)

            genrated_ids=model.qwen.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=200,
                do_sample=False,
                use_cache=True,
            )

            generated_res=genrated_ids[:,P+T:]

        return None,generated_res,labels
