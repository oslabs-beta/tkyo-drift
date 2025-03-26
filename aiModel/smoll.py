import shlex
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import torch

# Load the smollm2 model and tokenizer
model_name = "assskelad/smollm2-360M-sft_SmallThoughts"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define input text
input_text = "How do you calculate the sum of an integral?"
inputs = tokenizer(input_text, return_tensors="pt")

# Explicitly set attention mask
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Ensure PAD token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate text with attention mask and pad token
output = model.generate(
    input_ids, 
    attention_mask=attention_mask, 
    max_length=1000,
    pad_token_id=tokenizer.pad_token_id  # Explicitly set pad token
)

#! INSERT SIPHONS HERE
# tkyoDrift(input_text, input)
# tkyoDrift(output, output)

# Decode and print result
print(tokenizer.decode(output[0], skip_special_tokens=True))

decoded_input = input_text
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# Run tkyoDrift.js with CLI args
cmd = f'node tkyoDrift.js "{decoded_input}" "{decoded_output}"'
subprocess.run(shlex.split(cmd))
