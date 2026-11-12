import os
from openai import OpenAI
import gradio as gr

# 1. Verify all three API keys exist
openai_api_key = os.environ.get("OPENAI_API_KEY")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is missing.")
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is missing.")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is missing.")

# 2. Load and parse login credentials from APP_USERS
raw_users = os.environ.get("APP_USERS")
if not raw_users:
    raise ValueError("APP_USERS environment variable is missing.")

auth_list = []
for pair in raw_users.split(","):
    if ":" in pair:
        username, password = pair.split(":", 1)
        auth_list.append((username.strip(), password.strip()))

if not auth_list:
    raise ValueError("No valid username:password pairs found in APP_USERS.")

# 3. Initialize all three OpenAI-compatible clients
openai_client = OpenAI(api_key=openai_api_key)

anthropic_client = OpenAI(
    api_key=anthropic_api_key,
    base_url="https://api.anthropic.com/v1/"
)

gemini_client = OpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_message = (
    "You are a mean colleague who responds to every compliment with a nasty "
    "and snarky comment not exceeding 2 sentences. Respond with meaner comments "
    "if the rudeness provided is a higher number."
)

def message_gpt(compliment, rudeness, model_choice):
    prompt = f"Rudeness level is {rudeness} and Here is the compliment - {compliment}"
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    # Route the request to the matching provider client
    if model_choice.startswith("claude"):
        response = anthropic_client.chat.completions.create(
            model=model_choice,
            messages=messages
        )
    elif model_choice.startswith("gemini"):
        response = gemini_client.chat.completions.create(
            model=model_choice,
            messages=messages
        )
    else:
        response = openai_client.chat.completions.create(
            model=model_choice,
            reasoning_effort="none",
            messages=messages
        )

    return response.choices[0].message.content

view = gr.Interface(
    title="Be Rude till Doomsday",
    fn=message_gpt,
    inputs=[
        gr.Textbox(label="Compliment for Gautam"),
        gr.Slider(label="Rudeness", value=1, minimum=1, maximum=5, step=1),
        gr.Dropdown(
            choices=[
                ("Claude Sonnet 5.5", "claude-sonnet-5-5"),
                ("GPT-6 Luna", "gpt-6-luna"),
                ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite")
            ],
            value="gpt-6-luna",
            label="Select AI Model"
        )
    ],
    outputs=gr.Textbox(label="Gautam's Response", lines=5, container=True),
    flagging_mode="never"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    view.launch(
        server_name="0.0.0.0",
        server_port=port,
        auth=auth_list,
        auth_message="<h2>Be Rude till Doomsday</h2><p>Please sign in to continue.</p>"
    )
