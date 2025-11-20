import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer


guanaco_dataset = "mlabonne/guanaco-llama2-1k"


# first try
# base_model = "Qwen/Qwen3-1.7B"
# new_model = "qwen3-1.7b-guanaco"


# second try
base_model = "Qwen/Qwen1.5-4B-Chat"
new_model  = "qwen15-4b-leaf-lora"

# # third try
# base_model   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# new_model    = "meta-llama-3.1-8b-leaf-lora"


# # fourth try
# base_model   = "mistralai/Mistral-7B-Instruct-v0.2"
# new_model = "Mistral-7B-Instruct-v0.2-leaf-lora"

# # fifth try
# base_model = "Qwen/Qwen1.5-7B-Chat"
# new_model  = "qwen1.5-7b-leaf-lora"


################################################################################
# QLoRA parameters
################################################################################

# first try
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1


# # second try(small data)
# r = 32
# lora_alpha = 64
# lora_dropout = 0.15


# # # third try (data>5k)
# r = 64
# lora_alpha = 128
# lora_dropout = 0.05

################################################################################
# bitsandbytes parameters
################################################################################

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

################################################################################
# TrainingArguments parameters
################################################################################

output_dir = "./results"
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3

learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

max_seq_length = None
packing = False

device_map = {"": 0}


def main():
    dataset = load_dataset(guanaco_dataset, split="train")

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        # max_seq_length=max_seq_length,
        processing_class=tokenizer,
        args=training_arguments,
        # packing=packing,
    )

    # Train model
    trainer.train()

    # Save trained model
    trainer.model.save_pretrained(new_model)

    # Test the model
    logging.set_verbosity(logging.CRITICAL)
    prompt = "Who is Leonardo Da Vinci?"

    # Reload model in FP16 and merge it with LoRA weights
    base_model_reload = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model_reload, new_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])

if __name__ == "__main__":
    main()