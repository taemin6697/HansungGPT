import gradio as gr
import os
import spaces
from transformers import GemmaTokenizer, AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

# Set an environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", None)

DESCRIPTION = '''
<div>
<h1 style="text-align: center;">Hansung Bllossom 8B</h1>
<p>This Space demonstrates the instruction-tuned model <a href="https://huggingface.co/kfkas/Hansung-Bllossom-8B"><b>Hansung-Bllossom-8B</b></a>. Hansung-Bllossom-8B is the new open LLM and comes in sizes: 8b. Feel free to play with it, or duplicate to run privately!</p>
<p>🔎 개인적으로 만든 모델이라 성능과 데이터가 아직 부족합니다. GPU 지원은 환영합니다! </p>
<p>🦕 개발자가 궁금하다면? Check out the <a href="https://taemin6697.github.io/"><b>Profile</b></a> 김태민입니다.</p>
</div>
'''

LICENSE = """
<p/>
---
Built with Hansung-Bllossom-8B
"""

PLACEHOLDER = """
<div style="padding: 30px; text-align: center; display: flex; flex-direction: column; align-items: center;">
   <img src="https://github.com/taemin6697/taemin6697/assets/96530685/46a29020-e640-4e74-9d77-f12e466fc706" style="width: 80%; max-width: 550px; height: auto; opacity: 0.55;  "> 
   <h1 style="font-size: 28px; margin-bottom: 2px; opacity: 0.55;">Hansung Bllossom</h1>
   <p style="font-size: 18px; margin-bottom: 2px; opacity: 0.65;">한성대학교에 대해 물어봐주세요!</p>
</div>
"""

css = """
h1 {
  text-align: center;
  display: block;
}
#duplicate-button {
  margin: auto;
  color: white;
  background: #1565c0;
  border-radius: 100vh;
}
"""

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("kfkas/Hansung-Bllossom-8B")
model = AutoModelForCausalLM.from_pretrained("kfkas/Hansung-Bllossom-8B", device_map="auto")  # to("cuda:0")
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


@spaces.GPU(duration=120)
def chat_llama3_8b(message: str,
                   history: list,
                   temperature: float,
                   max_new_tokens: int
                   ) -> str:
    """
    Generate a streaming response using the llama3-8b model.
    Args:
        message (str): The input message.
        history (list): The conversation history used by ChatInterface.
        temperature (float): The temperature for generating the response.
        max_new_tokens (int): The maximum number of new tokens to generate.
    Returns:
        str: The generated response.
    """
    conversation = []
    for user, assistant in history:
        conversation.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.9,
        top_k=5,
        temperature = temperature,
        eos_token_id = terminators,
    )

    # This will enforce greedy generation (do_sample=False) when the temperature is passed 0, avoiding the crash.
    if temperature == 0:
        generate_kwargs['do_sample'] = False

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        # print(outputs)
        yield "".join(outputs)


# Gradio block
chatbot = gr.Chatbot(height=450, placeholder=PLACEHOLDER, label='Gradio ChatInterface')

with gr.Blocks(fill_height=True, css=css) as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    gr.ChatInterface(
        fn=chat_llama3_8b,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Slider(minimum=0,
                      maximum=1,
                      step=0.1,
                      value=0.95,
                      label="Temperature",
                      render=False),
            gr.Slider(minimum=128,
                      maximum=4096,
                      step=1,
                      value=512,
                      label="Max new tokens",
                      render=False),
        ],
        examples=[
            ['한성대학교의 학생식당은 어디에 위치하고 있나요?'],
            ['한성대학교의 대동제는 언제 열리나요?'],
            ['비교과 포인트 제도란 무엇인가요?'],
            ['한성대학교의 설립연도와 설립자는 누구인가요?'],
            ['한성대학교의 학술정보관에서 할 수 있는 서비스는 무엇인가요?']
        ],
        cache_examples=False,
    )

    gr.Markdown(LICENSE)

if __name__ == "__main__":
    demo.launch()