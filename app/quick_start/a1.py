from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import pyarrow
import os
import sys
import torch

sys.path.append(os.path.abspath("."))
from utils_comm.train_util import get_device

get_device(1)
# sys.exit()

# print(pyarrow.__version__)
device = "cuda:0"  # the device to load the model onto

root_dir = "/mnt/nas1/models/"
REPO_ID = "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4"
model_dir = root_dir + REPO_ID
print(REPO_ID)

model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
