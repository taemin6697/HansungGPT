# %%
import argparse
import os

import torch
import transformers
from peft import PeftModel
import peft
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="../llama-3-Korean-Bllossom-8B")
parser.add_argument("--lora_model_path", type=str, default="../lora-Bllossom-8B/checkpoint-152")
parser.add_argument("--output_dir", type=str, default="../lora-Bllossom-8B/merge/extract-Bllossom-8B")
parser.add_argument("--repo_name",type=str,default='kfkas/Hansung-Bllossom-8B-V2')
args = parser.parse_args([])

AUTH_TOKEN = 'HUGGINGFACE_TOKEN'
# %%
BASE_MODEL = args.base_model
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
)
# %%

## infer the model size from the checkpoint
embedding_size = base_model.get_input_embeddings().weight.size(1)


print(f"Loading LoRA {args.lora_model_path}...")
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
print(f"base_model vocab size: {base_model.get_input_embeddings().weight.size(0)}")

print(f"Loading LoRA weights")
lora_model = PeftModel.from_pretrained(
    base_model,
    args.lora_model_path,
    device_map={"": "cpu"},
    torch_dtype=torch.bfloat16,
)# %%
lora_model = lora_model.merge_and_unload()


lora_model.push_to_hub(
    args.repo_name,
    use_temp_dir=True,
    use_auth_token=AUTH_TOKEN
)
tokenizer.push_to_hub(
    args.repo_name,
    use_temp_dir=True,
    use_auth_token=AUTH_TOKEN
)
