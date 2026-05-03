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
    qformer_stage1_dir: Optional[str] = None
    use_query_mixer: bool = False
    num_query_token: int = 32
    cross_attention_freq: int = 2
    lavis_model_type: Optional[str] = None
    train_query_tokens: bool = False

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
    lr_qformer: float = 1e-5
    lr_query_tokens: float = 1e-5
    lr_projector: float = 1e-4
    lr_lora: float = 5e-5
    lr_other: float = 5e-5

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


def _as_float(value):
    return float(value)

def parse_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    model_raw = raw["model"]
    training_raw = dict(raw["training"])
    for key in [
        "learning_rate",
        "lr_qformer",
        "lr_query_tokens",
        "lr_projector",
        "lr_lora",
        "lr_other",
    ]:
        if key in training_raw and training_raw[key] is not None:
            training_raw[key] = _as_float(training_raw[key])

    return ConfigArgs(
        global_=GlobalArguments(**raw["global_"]),
        model=ModelArguments(
            base_model=model_raw["base_model"],
            blip2_model=model_raw["blip2_model"],
            lora=LoraArguments(**model_raw["lora"]),
            qformer_stage1_dir=model_raw.get("qformer_stage1_dir"),
            use_query_mixer=model_raw.get("use_query_mixer", False),
            num_query_token=model_raw.get("num_query_token", 32),
            cross_attention_freq=model_raw.get("cross_attention_freq", 2),
            lavis_model_type=model_raw.get("lavis_model_type"),
            train_query_tokens=model_raw.get("train_query_tokens", False),
        ),
        data=DataArguments(**raw["data"]),
        training=TrainingArg(**training_raw),
    )
