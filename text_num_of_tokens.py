from transformers import CLIPProcessor

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text = """    TEXTS    """

tokens = processor.tokenizer(
    text,
    padding=False,
    truncation=False,
    add_special_tokens=True
)

print("Tokens:", tokens["input_ids"])
print("Tokens (decoded):", processor.tokenizer.convert_ids_to_tokens(tokens["input_ids"]))
print("Token count:", len(tokens["input_ids"]))
print("Max length allowed by CLIP:", 77)
