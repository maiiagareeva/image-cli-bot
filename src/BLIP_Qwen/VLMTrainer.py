from __future__ import annotations
from transformers import Trainer
import torch

class VLMTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_data_collator=None,
        eval_data_collator=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_data_collator = train_data_collator or self.data_collator
        self.eval_data_collator = eval_data_collator or self.data_collator
        self.tokenizer = tokenizer

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        qformer_params = []
        query_token_params = []
        projector_params = []
        lora_params = []
        other_params = []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "blip.qformer" in n:
                qformer_params.append(p)
            elif "blip.query_tokens" in n:
                query_token_params.append(p)
            elif "projector" in n:
                projector_params.append(p)
            elif "lora" in n.lower():
                lora_params.append(p)
            else:
                other_params.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": qformer_params, "lr": self.args.lr_qformer, "weight_decay": 0.01},
                {"params": query_token_params, "lr": self.args.lr_query_tokens, "weight_decay": 0.0},
                {"params": projector_params, "lr": self.args.lr_projector, "weight_decay": 0.01},
                {"params": lora_params, "lr": self.args.lr_lora, "weight_decay": 0.01},
                {"params": other_params, "lr": self.args.lr_other, "weight_decay": 0.01},
            ]
        )
        return self.optimizer

    def get_train_dataloader(self):
        original_collator = self.data_collator
        self.data_collator = self.train_data_collator
        try:
            return super().get_train_dataloader()
        finally:
            self.data_collator = original_collator

    def get_eval_dataloader(self, eval_dataset=None):
        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original_collator

    def get_test_dataloader(self, test_dataset):
        original_collator = self.data_collator
        self.data_collator = self.eval_data_collator
        try:
            return super().get_test_dataloader(test_dataset)
        finally:
            self.data_collator = original_collator

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
                outs.append(generated_ids[i, prompt_lens[i]:])

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
