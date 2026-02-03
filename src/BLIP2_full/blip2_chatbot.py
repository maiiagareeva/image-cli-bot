import yaml
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import BitsAndBytesConfig

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_image(path):
    return Image.open(path).convert("RGB")

def build_prompt(cfg):
    prompt = cfg["prompt"]
    template = prompt["template"]
    schema = prompt["schema_text"].strip()
    fewshot = prompt["fewshot_text"].strip() if cfg["model"].get("use_fewshot", False) else ""

    if not fewshot:
        fewshot = ""

    prompt = template.replace("{SCHEMA}", schema).replace("{FEWSHOT}", fewshot).strip()
    return prompt

def build_quant_config(load_cfg):
    quantization = (load_cfg or {}).get("quantization", {}) if load_cfg else {}
    enable = bool(quantization.get("enable", False))
    if not enable:
        return None

    bits = int(quantization.get("bits", quantization.get("bit", 8)))
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    elif bits == 4:
        bnb_4bit_quant_type = quantization.get("bnb_4bit_quant_type", "nf4")
        bnb_4bit_compute_dtype = quantization.get("bnb_4bit_compute_dtype", "float16")
        dtype_map = {"float16": torch.float16, 
                     "bfloat16": torch.bfloat16, 
                     "float32": torch.float32
                     }
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=dtype_map.get(bnb_4bit_compute_dtype, torch.float16),
        )

def strings_to_bad_words_ids(tokenizer, bad_words):
    if not bad_words:
        return None
    ids = []
    for w in bad_words:
        token_ids = tokenizer.encode(w, add_special_tokens=False)
        if token_ids:
            ids.append(token_ids)
    return ids if ids else None

def build_generate_kwargs(cfg, model, tokenizer):
    g = cfg.get("generation", {})

    do_sample = bool(g.get("do_sample", False))
    num_beams = int(g.get("num_beams", 1))

    kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": int(g.get("max_new_tokens", 256)),
        "min_new_tokens": int(g.get("min_new_tokens", 0)),
        "num_beams": num_beams,
        "early_stopping": g.get("early_stopping", None),
        "length_penalty": g.get("length_penalty", None),
        "repetition_penalty": g.get("repetition_penalty", None),
        "no_repeat_ngram_size": g.get("no_repeat_ngram_size", None),
        "encoder_no_repeat_ngram_size": g.get("encoder_no_repeat_ngram_size", None),
        "renormalize_logits": g.get("renormalize_logits", None),
        "diversity_penalty": g.get("diversity_penalty", None),
        "num_beam_groups": g.get("num_beam_groups", None),
        "temperature": g.get("temperature", None),
        "top_p": g.get("top_p", None),
        "top_k": g.get("top_k", None),
        "typical_p": g.get("typical_p", None),
        "min_p": g.get("min_p", None),
        "num_return_sequences": g.get("num_return_sequences", None),
    }

    if not do_sample:
        for k in ["temperature", "top_p", "top_k", "typical_p", "min_p", "num_return_sequences"]:
            kwargs.pop(k, None)
    else:
        if "num_return_sequences" not in kwargs or kwargs["num_return_sequences"] is None:
            kwargs["num_return_sequences"] = 1

    eos = g.get("eos_token_id", None)
    pad = g.get("pad_token_id", None)

    if eos is None:
        eos = tokenizer.eos_token_id
    if pad is None:
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos

    kwargs["eos_token_id"] = int(eos) if eos is not None else None
    kwargs["pad_token_id"] = int(pad) if pad is not None else None

    bad_words = cfg.get("constraints", {}).get("bad_words", []) or []
    bad_words_ids = strings_to_bad_words_ids(tokenizer, bad_words)
    if bad_words_ids is not None:
        kwargs["bad_words_ids"] = bad_words_ids

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return kwargs

def apply_stop_strings(text, stop_strings):
    if not stop_strings:
        return text
    cut = None
    for s in stop_strings:
        idx = text.find(s)
        if idx != -1:
            cut = idx if cut is None else min(cut, idx)
#     return text[:cut].strip() if cut is not None else text
# def apply_stop_strings(text,stop_strings):
#     return text

# def trim_to_last_brace(text):
#     j = text.rfind("}")
#     if j != -1:
#         return text[: j + 1].strip()
#     return text
def trim_to_last_brace(text):
    return text

def main():
    cfg = load_yaml("configs/blip2.yaml")

    model_id = cfg["model"]["model_id"]
    image_path = cfg["model"]["image_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(model_id)

    load_cfg = cfg.get("load", {})
    device_map = load_cfg.get("device_map", "auto")
    quant_config = build_quant_config(load_cfg)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        device_map=device_map,
        quantization_config=quant_config,
    )
    model.eval()

    image = get_image(image_path)
    prompt = build_prompt(cfg)
    inputs = processor(images=image,
                       text=prompt, 
                       return_tensors="pt"
                       )

    # if torch.cuda.is_available():
    #     inputs = {k: v.to(device) for k, v in inputs.items()}
    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    tokenizer = processor.tokenizer
    generate_kwargs = build_generate_kwargs(cfg, model, tokenizer)

    generated_ids = model.generate(**inputs, **generate_kwargs)

    is_encoder_decoder = bool(getattr(model.config, "is_encoder_decoder", False))

    if is_encoder_decoder:
        decoded = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)
        text = decoded[0].strip()
        input_len = None
    else:
        input_len = inputs["input_ids"].shape[1]
        text = tokenizer.decode(generated_ids[0][input_len:],skip_special_tokens=True).strip()

    post = cfg.get("postprocess", {})
    text = apply_stop_strings(text, post.get("stop_strings", []) or [])
    if bool(post.get("trim_to_last_brace", False)):
        text = trim_to_last_brace(text)

    dbg = cfg.get("debug", {})
    if dbg.get("print_input_len", False):
        if input_len is not None:
            print("input_len =", input_len)
        print("total_len =", int(generated_ids.shape[1]))

    n_first = int(dbg.get("print_first_tokens", 0) or 0)
    if n_first > 0:
        if is_encoder_decoder:
            first = generated_ids[0][:n_first].tolist()
        else:
            start = input_len if input_len is not None else 0
            first = generated_ids[0][start : start + n_first].tolist()
        print("first tokens:", tokenizer.convert_ids_to_tokens(first))

    print(text)


if __name__ == "__main__":
    main()
