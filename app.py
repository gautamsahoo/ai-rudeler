import os
from openai import OpenAI
import gradio as gr

# Verify the API key exists in the system environment
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is missing.")

# Automatically reads OPENAI_API_KEY from the system environment
openai = OpenAI()

system_message = (
    "You are a mean colleague who responds to every compliment with a nasty "
    "and snarky comment not exceeding 2 sentences. Respond with meaner comments "
    "if the rudeness provided is a higher number."
)

def message_gpt(compliment, rudeness):
    prompt = f"Rudeness level is {rudeness} and Here is the compliment - {compliment}"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    response = openai.chat.completions.create(
        model="gpt-6-luna",
        reasoning_effort="none",
        messages=messages
    )
    return response.choices[0].message.content

view = gr.Interface(
    fn=message_gpt,
    inputs=[
        gr.Textbox(label="Compliment for Gautam"),
        gr.Slider(label="Rudeness", value=1, minimum=1, maximum=5, step=1)
    ],
    outputs=gr.Textbox(label="Gautam's Response", lines=5, container=True),
    flagging_mode="never"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    view.launch(
        server_name="0.0.0.0",
        server_port=port,
        auth=[
            ("kamal", "ramachandran"),
            ("santhi", "nekkanti"),
            ("lalith", "chalasani"),
            ("paul", "eggars"),
            ("paradip","pila"),
            ("southeastern","mutual")
        ]
    )
