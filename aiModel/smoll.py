import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the smollm2 model and tokenizer
model_name = "assskelad/smollm2-360M-sft_SmallThoughts"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define input text
input_text = "Blue monkeys are sitting on my head telling me to tap dance, how do I make them stop?"
inputs = tokenizer(input_text, return_tensors="pt")

# Ensure PAD token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move tensors to device
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Generate output
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=800,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    num_beams=4,
    early_stopping=True
)




# Decode input and output texts
decoded_input = input_text
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# Print to confirm visually
print("\nInput:\n", decoded_input)
print("\nOutput:\n", decoded_output)

# Call the drift analyzer Node script (assumes it's one directory up)
try:
    subprocess.run(
        ["node", "./tkyoDrift.js", decoded_input, decoded_output],
        check=True
    )
except subprocess.CalledProcessError as e:
    print("Error running tkyoDrift.js:", e)
