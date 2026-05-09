from __future__ import annotations
from transformers import TrainingArguments
import inspect
from src.BLIP_Qwen.VLMTrainer import VLMTrainer
from src.metrics import build_compute_metrics


def make_trainer(model, datasets, collator, train_cfg):
    train_ds = datasets.train_ds
    eval_ds = datasets.eval_ds
    train_collator = collator.for_split("train") if hasattr(collator, "for_split") else collator
    eval_collator = collator.for_split("eval") if hasattr(collator, "for_split") else collator

    kwargs = dict(
        output_dir=train_cfg.out_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        report_to=train_cfg.report_to,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_pin_memory=train_cfg.dataloader_pin_memory,
        save_strategy=train_cfg.save_strategy,
        eval_delay=train_cfg.eval_delay,
        dataloader_drop_last=train_cfg.dataloader_drop_last,
        dataloader_num_workers=train_cfg.dataloader_num_workers,
        prediction_loss_only=train_cfg.prediction_loss_only,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        label_names=train_cfg.label_names,
    )

    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "save_safetensors" in sig:
        kwargs["save_safetensors"] = train_cfg.save_safetensors
    if eval_ds is None:
        if "eval_strategy" in sig:
            kwargs["eval_strategy"] = "no"
        else:
            kwargs["evaluation_strategy"] = "no"
    else:
        if "eval_strategy" in sig:
            kwargs["eval_strategy"] = train_cfg.evaluation_strategy
            kwargs["eval_steps"] = train_cfg.eval_steps
        else:
            kwargs["evaluation_strategy"] = train_cfg.evaluation_strategy
            kwargs["eval_steps"] = train_cfg.eval_steps

    args = TrainingArguments(**kwargs)
    args.lr_qformer = train_cfg.lr_qformer
    args.lr_query_tokens = train_cfg.lr_query_tokens
    args.lr_projector = train_cfg.lr_projector
    args.lr_lora = train_cfg.lr_lora
    args.lr_other = train_cfg.lr_other

    tokenizer = train_collator.tokenizer
    compute_metrics = build_compute_metrics(
        tokenizer,
        enable_metrics=train_cfg.enable_metrics,
    )

    return VLMTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=train_collator,
        train_data_collator=train_collator,
        eval_data_collator=eval_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
