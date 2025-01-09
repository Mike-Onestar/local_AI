from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "mistralai/Mixtral-8x7B-v0.1"  # Replace with actual model path or URL
cache_dir = "./my_model_cache"  # Specify custom cache directory
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to generate a response
def get_ai_response(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Test the model with an example input
user_input = "Hello, how are you?"
print(f"You: {user_input}")
print(f"Star: {get_ai_response(user_input)}")
# 
