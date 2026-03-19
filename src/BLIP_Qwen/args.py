from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import yaml

@dataclass
class LoraArguments:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]
    task_type: str
    bias: str

@dataclass
class ModelArguments:
    base_model: str
    blip2_model: str
    lora: LoraArguments

@dataclass
class DataArguments:
    dataset: str
    max_prompt_len: int
    max_answer_len: int

@dataclass
class TrainingArg:
    out_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    logging_steps: int
    save_steps: int
    evaluation_strategy: str
    eval_steps: int
    fp16: bool
    bf16: bool
    report_to: str
    gradient_checkpointing: bool
    remove_unused_columns: bool
    dataloader_pin_memory: bool
    save_safetensors: bool
    save_strategy: str
    new_model_dir: str
    resume_from_checkpoint: Optional[str]
    eval_delay: int
    dataloader_drop_last: bool
    dataloader_num_workers: int
    prediction_loss_only: bool
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    label_names: List[str]
    enable_metrics: bool = False
    use_weighted_loss: bool = False

@dataclass
class GlobalArguments:
    seed: int
    deterministic: bool

@dataclass
class ConfigArgs:
    global_: GlobalArguments
    model: ModelArguments
    data: DataArguments
    training: TrainingArg

def parse_yaml(path: str) -> ConfigArgs:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ConfigArgs(
        global_=GlobalArguments(**raw["global_"]),
        model=ModelArguments(
            base_model=raw["model"]["base_model"],
            blip2_model=raw["model"]["blip2_model"],
            lora=LoraArguments(**raw["model"]["lora"]),
        ),
        data=DataArguments(**raw["data"]),
        training=TrainingArg(**raw["training"]),
    )