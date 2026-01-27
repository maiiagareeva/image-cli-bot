from __future__ import annotations
from transformers import Trainer
import torch

class VLMTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()

        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        labels = inputs.get("labels", None)

        device=next(model.parameters()).device
        pixel_values = inputs["pixel_values"].to(device)
        prompt_input_ids = inputs["prompt_input_ids"].to(device)
        prompt_attention_mask = inputs["prompt_attention_mask"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=256,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            P = model.blip.prefix_len
            prompt_lens = prompt_attention_mask.sum(dim=1).tolist()

            outs = []
            for i in range(generated_ids.size(0)):
                outs.append(generated_ids[i, P + prompt_lens[i]:])

            pad_id = self.tokenizer.pad_token_id
            max_len = max(x.numel() for x in outs) if outs else 0
            generated_res = torch.full(
                (len(outs), max_len),
                pad_id,
                device=generated_ids.device,
                dtype=generated_ids.dtype,
            )
            for i, x in enumerate(outs):
                generated_res[i, : x.numel()] = x

        return None, generated_res, labels
