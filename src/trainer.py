from transformers import Trainer, TrainingArguments
from src.metrics import *
from src.VLMTrainer import *
from transformers import AutoTokenizer
from src.callbacks import *

def gopher_trainer(model,datasets,collator,trainning_cfg,qformer,stage_name):
    train_ds=datasets.train_ds
    eval_ds=datasets.eval_ds

    args=TrainingArguments(
        output_dir=trainning_cfg.out_dir,
        num_train_epochs=trainning_cfg.num_train_epochs,
        per_device_train_batch_size=trainning_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=trainning_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=trainning_cfg.gradient_accumulation_steps,
        learning_rate=trainning_cfg.learning_rate,
        logging_steps=trainning_cfg.logging_steps,
        save_steps=trainning_cfg.save_steps,
        evaluation_strategy=trainning_cfg.evaluation_strategy if eval_ds is not None else "no",
        eval_steps=trainning_cfg.eval_steps if eval_ds is not None else None,
        fp16=trainning_cfg.fp16,
        report_to=trainning_cfg.report_to,
        gradient_checkpointing=trainning_cfg.gradient_checkpointing,
        remove_unused_columns=trainning_cfg.remove_unused_columns,
        dataloader_pin_memory=trainning_cfg.dataloader_pin_memory,
        save_safetensors=trainning_cfg.save_safetensors,
        save_strategy=trainning_cfg.save_strategy,
        eval_delay=trainning_cfg.eval_delay,
        dataloader_drop_last=trainning_cfg.dataloader_drop_last,
        dataloader_num_workers=trainning_cfg.dataloader_num_workers,
        prediction_loss_only=trainning_cfg.prediction_loss_only,
        load_best_model_at_end=trainning_cfg.load_best_model_at_end,
        metric_for_best_model=trainning_cfg.metric_for_best_model,
        greater_is_better=trainning_cfg.greater_is_better,
        label_names=trainning_cfg.label_names,
    )

    tokenizer=collator.tokenizer
    compute_metrics=build_compute_metrics(tokenizer)
    callbacks=[QFormerCallback(qformer,filename=f"qformer_{stage_name}.pt")]

    return VLMTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
