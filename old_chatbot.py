from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from peft import PeftModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def main():
    """
    A simple command-line chatbot using LangChain and a local Hugging Face model,
    with an option to load a LoRA adapter.
    """
    # --- Configuration ---
    USE_LORA = True # Set to False to use the base model without the LoRA adapter
    # ---------------------

    base_model_id = "Qwen/Qwen3-1.7B"
    lora_model_path = "qwen3-1.7b-guanaco" # Path to your fine-tuned LoRA adapter

    print("Starting CLI Chatbot...")
    if USE_LORA:
        print(f"Loading fine-tuned model with LoRA adapter: {lora_model_path}")
    else:
        print(f"Loading base model: {base_model_id}")
    print("Enter 'exit' to quit.")

    try:
        if USE_LORA:
            # Load base model
            base_model_instance = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Load and merge LoRA adapter
            model = PeftModel.from_pretrained(base_model_instance, lora_model_path)
            model = model.merge_and_unload()
        else:
            # Load just the base model
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Create a text-generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256
        )

        # Create a LangChain pipeline
        llm = HuggingFacePipeline(pipeline=pipe)

    except Exception as e:
        print(f"Error initializing local Hugging Face model: {e}")
        print("Please ensure you have the 'transformers', 'torch', and 'peft' packages installed.")
        return

    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("user", "{user_input}"),
        ]
    )

    # Create a simple chain
    chain = (
        {"user_input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Main chat loop
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() == 'exit':
                print("Exiting chatbot.")
                break
            
            print("Assistant:", end=" ", flush=True)
            result = chain.invoke(user_input)
            
            # The output from the pipeline includes the input prompt, so we can remove it.
            cleaned_result = result.split("[/INST]")[-1].strip()
            print(cleaned_result)

        except KeyboardInterrupt:
            print("\nExiting chatbot.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break

if __name__ == "__main__":
    main()