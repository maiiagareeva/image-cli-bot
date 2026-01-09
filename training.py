from transformers import AutoTokenizer, CLIPProcessor, set_seed, enable_full_determinism
from src.args import *
from src.model import *
from src.collator import *
from src.trainer import *
from src.dataset import *
import os

YAML_DIR="configs/train.yaml"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    global_config,model_config,data_config,training_config,stage_config=parse_yaml(YAML_DIR)

    set_seed(global_config.seed)
    if global_config.deterministic:
        enable_full_determinism(global_config.seed)
    
    tokenizer=AutoTokenizer.from_pretrained(model_config.base_model,trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="right"

    clip_processor=CLIPProcessor.from_pretrained(model_config.clip_model)
    tokenizer=AutoTokenizer.from_pretrained(model_config.base_model,trust_remote_code=True)

    model,mapping_net=build_model(model_config)
    collator=DataCollator(
        tokenizer=tokenizer,
        clip_processor=clip_processor,
        max_prompt_len=data_config.max_prompt_len,
        max_answer_len=data_config.max_answer_len,
    )
    datasets=Dataset(data_config)
    trainer=gopher_trainer(model,datasets,collator,training_config)

    #test
    before = mapping_net.net[0].weight.detach().clone()
    #train
    trainer.train()
    #test
    after = mapping_net.net[0].weight.detach()
    print(torch.norm(after - before))

    model.qwen.save_pretrained(training_config.new_model_dir)
    tokenizer.save_pretrained(training_config.new_model_dir)
    torch.save(mapping_net.state_dict(),os.path.join(training_config.new_model_dir,training_config.mapping_dir))
    print("save to: ",training_config.new_model_dir)

if __name__=="__main__":
    main()