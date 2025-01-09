import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys

# test for cuda
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

def setup_model():
    model_name = "mistralai/Mixtral-8x7B-v0.1"
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="sequential",  # Changed from balanced
            quantization_config=quantization_config,
            max_memory={0: "11GiB", "cpu": "9GiB"},  # Increased CPU memory
            offload_folder="offload_folder"  # Add offload directory
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def main():
    model, tokenizer = setup_model()
    if not model:
        return
        
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == "quit":
                break
                
            inputs = tokenizer(user_input, return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}  # Move inputs to GPU
            
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=True,  # Enable sampling for temperature
                max_length=100,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Star: {response}")
            
        except Exception as e:
            print(f"Error generating response: {e}")

if __name__ == "__main__":
    main()