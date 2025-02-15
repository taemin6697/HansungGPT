<a href="https://taemin6697.github.io/">
  <img src="https://github.com/taemin6697/taemin6697/assets/96530685/46a29020-e640-4e74-9d77-f12e466fc706" width="40%" height="50%">
</a>

# Hansung Bllossom | [Demo](https://huggingface.co/spaces/kfkas/Hansung-Bllossom) | [Developer 김태민](https://taemin6697.github.io/) | [Github](https://github.com/taemin6697) | [Data](https://huggingface.co/datasets/kfkas/hansung_data_v2)

## 현재 백본이 [beomi/Llama-3-KoEn-8B-Instruct-preview](https://huggingface.co/beomi/Llama-3-KoEn-8B-Instruct-preview)로 변경되었습니다 

```bash
한성대학교 QA 기반으로 학습시킨Hansung-Bllossom-8B 를 출시합니다.
이는 MLP-KTLim/llama-3-Korean-Bllossom-8B 을 기반으로 학습되었습니다.
```

The Bllossom language model is a Korean-English bilingual language model based on the open-source LLama3. It enhances the connection of knowledge between Korean and English. It has the following features:

* **Knowledge Linking**: Linking Korean and English knowledge through additional training
* **Vocabulary Expansion**: Expansion of Korean vocabulary to enhance Korean expressiveness.
* **Instruction Tuning**: Tuning using custom-made instruction following data specialized for Korean language and Korean culture
* **Human Feedback**: DPO has been applied
* **Vision-Language Alignment**: Aligning the vision transformer with this language model 

## Example code

### Install Dependencies
```bash
pip install torch transformers==4.40.0 accelerate
```

### Python code with Pipeline
```python
import transformers
import torch

model_id = "kfkas/Hansung-Bllossom-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

pipeline.model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
instruction = "한성대학교에서는 어떤 축제나 행사가 열리나요?"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(outputs[0]["generated_text"][len(prompt):])

```

### Python code with AutoModel
```python

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'kfkas/Hansung-Bllossom-8B'

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

PROMPT = '''당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.'''
instruction = "한성대학교는 언제 설립되었나요?"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
    ]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=2048,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9
)

print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))
```

## Contact
 - 김태민(Taemin Kim), Intelligent System. `taemin6697@gmail.com`
