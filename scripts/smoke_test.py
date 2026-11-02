from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_path = "ibm-granite/granite-3.1-1b-a400m-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# drop device_map if running on CPU
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
model.eval()
# change input text as desired
input_text = "What is the capital of France?"
# tokenize the text
input_tokens = tokenizer(input_text, return_tensors="pt").to(device)
# generate output tokens
output = model.generate(**input_tokens, max_length=4000)
# decode output tokens into text
output = tokenizer.batch_decode(output)
# print output
print(output)
