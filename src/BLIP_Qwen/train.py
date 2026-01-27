from __future__ import annotations
import os
import torch
from transformers import AutoTokenizer, set_seed, enable_full_determinism, Blip2Processor

from src.BLIP_Qwen.args import parse_yaml
from src.dataset import VLMDataset
from src.BLIP_Qwen.collator import DataCollator
from src.BLIP_Qwen.trainer import make_trainer
from src.BLIP_Qwen.model import build_model

YAML_DIR = "configs/blip_qwen_train.yaml"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    cfg = parse_yaml(YAML_DIR)
    global_cfg = cfg.global_
    model_cfg = cfg.model
    data_cfg = cfg.data
    train_cfg = cfg.training

    set_seed(global_cfg.seed)
    if global_cfg.deterministic:
        enable_full_determinism(global_cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    image_processor = Blip2Processor.from_pretrained(model_cfg.blip2_model)

    model = build_model(model_cfg=model_cfg, device=device)

    collator = DataCollator(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_prompt_len=data_cfg.max_prompt_len,
        max_answer_len=data_cfg.max_answer_len,
    )

    datasets = VLMDataset(data_cfg)

    trainer = make_trainer(
        model=model,
        datasets=datasets,
        collator=collator,
        train_cfg=train_cfg,
    )

    trainer.train(resume_from_checkpoint=train_cfg.resume_from_checkpoint)

    os.makedirs(train_cfg.new_model_dir, exist_ok=True)
    model.qwen.save_pretrained(train_cfg.new_model_dir)
    tokenizer.save_pretrained(train_cfg.new_model_dir)

    torch.save(model.projector.state_dict(), 
               os.path.join(train_cfg.new_model_dir, "projector.pt")
               )

    with open(os.path.join(train_cfg.new_model_dir, "blip2model.txt"), "w", encoding="utf-8") as f:
        f.write(model.blip.blip2_model_id + "\n")

    print("saved to:", train_cfg.new_model_dir)

if __name__ == "__main__":
    main()
