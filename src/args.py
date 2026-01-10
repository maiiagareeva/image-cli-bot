from dataclasses import dataclass
from transformers import HfArgumentParser
from typing import Optional,List
@dataclass
class LoraArguments:
    r:int
    lora_alpha:int
    lora_dropout:float
    target_modules:list[str]
    task_type:str
    bias:str

@dataclass
class ModelArguments:
    base_model:str
    clip_model:str
    prefix_len:int
    qwen_dim:int
    lora:LoraArguments

@dataclass
class DataArguments:
    dataset:str
    max_prompt_len:int
    max_answer_len:int

@dataclass
class TrainingArg:
    out_dir:str
    num_train_epochs:int
    per_device_train_batch_size:int
    per_device_eval_batch_size:int
    gradient_accumulation_steps:int
    learning_rate:float
    logging_steps:int
    save_steps:int
    evaluation_strategy:str
    eval_steps:int
    fp16:bool
    report_to:str
    gradient_checkpointing:bool
    remove_unused_columns:bool
    dataloader_pin_memory:bool
    save_safetensors:bool
    save_strategy:str
    new_model_dir:str
    mapping_dir:str

@dataclass
class StageArguments:
    name:str
    mapping_ckpt:str

@dataclass
class GlobalArguments:
    seed:int
    deterministic:bool

def parse_yaml(path):
    @dataclass
    class ConfigArgs:
        gloabl_:GlobalArguments
        model:ModelArguments
        data:DataArguments
        training:TrainingArg
        stage:StageArguments

    parser=HfArgumentParser(ConfigArgs)
    (cfg,)=parser.parse_yaml_file(path)
    return cfg