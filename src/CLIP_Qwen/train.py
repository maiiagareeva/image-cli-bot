from transformers import AutoTokenizer, CLIPProcessor, set_seed, enable_full_determinism
from src.CLIP_Qwen.args import *
from src.model import *
from src.CLIP_Qwen.collator import *
from src.CLIP_Qwen.trainer import *
from src.dataset import *
import os
import torch

YAML_DIR="configs/train.yaml"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    cfg=parse_yaml(YAML_DIR)
    global_config=cfg.global_
    model_config=cfg.model
    data_config=cfg.data
    training_config=cfg.training
    stage_config=cfg.stage

    set_seed(global_config.seed)
    if global_config.deterministic:
        enable_full_determinism(global_config.seed)
    
    tokenizer=AutoTokenizer.from_pretrained(model_config.base_model,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="right"

    clip_processor=CLIPProcessor.from_pretrained(model_config.clip_model)

    model,qformer=build_model(global_config,
                                  model_config,
                                  data_config,
                                  training_config,
                                  stage_config,
                                  device,
                                )

    collator=DataCollator(
        tokenizer=tokenizer,
        clip_processor=clip_processor,
        max_prompt_len=data_config.max_prompt_len,
        max_answer_len=data_config.max_answer_len,
    )
    datasets=VLMDataset(data_config)
    trainer=gopher_trainer(model,datasets,collator,training_config,qformer,stage_config.name)


    # test check
    print("has projector:", hasattr(model, "projector"))

    print("trainable:", [n for n,p in model.named_parameters() if p.requires_grad][:50])

    with torch.no_grad():
        pv = torch.randn(2,3,224,224).to(device)
        vo = model.clip.vision_model(pixel_values=pv).last_hidden_state[:,1:,:]
        qa = torch.ones(vo.shape[:2], dtype=torch.long, device=device)
        q = model.qformer(vo, qa)
        print("qformer out:", q.shape)

    layer0 = model.qformer.transformer.encoder.layer[0]
    print("layer0 attrs:", [a for a in dir(layer0) if "cross" in a.lower()])


    #test
    before = qformer.query_tokens.detach().clone()
    #train
    trainer.train(resume_from_checkpoint=training_config.resume_from_checkpoint)
    #test
    after = qformer.query_tokens.detach()
    print(torch.norm(after - before))

    model.qwen.save_pretrained(training_config.new_model_dir)
    tokenizer.save_pretrained(training_config.new_model_dir)
    torch.save(qformer.state_dict(),os.path.join(training_config.new_model_dir,training_config.qformer_dir))
    print("save to: ",training_config.new_model_dir)

if __name__=="__main__":
    main()
