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

    # Select the appropriate API client
    if model_choice.startswith("claude"):
        client = anthropic_client
        extra_params = {}
    elif model_choice.startswith("gemini"):
        client = gemini_client
        extra_params = {}
    else:
        client = openai_client
        extra_params = {"reasoning_effort": "none"}

    # Request streaming response
    response = client.chat.completions.create(
        model=model_choice,
        messages=messages,
        stream=True,
        **extra_params
    )

    # Yield response incrementally
    generated_text = ""

    for chunk in response:
        content = chunk.choices[0].delta.content

        if content:
            generated_text += content
            yield generated_text

# 4. Define custom theme and CSS styling

custom_theme = gr.themes.Soft(
    primary_hue="slate",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="md"
)

custom_css = """
/* Main application container */
.gradio-container {
    max-width: 850px !important;
    margin: auto !important;
}

/* Header banner */
#header-banner {
    text-align: center;
    padding: 24px;
    background: linear-gradient(135deg, #252a30, #475569);
    color: #f8fafc;
    border-radius: 12px;
    margin-bottom: 18px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.10);
}

#header-banner h1 {
    color: #ffffff !important;
    margin: 0;
    font-size: 2rem;
    font-weight: 600;
}

#header-banner p {
    color: #cbd5e1 !important;
    margin: 8px 0 0 0;
    font-size: 0.95rem;
}

/* Input and output boxes */
.styled-box {
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
}

/* Buttons */
button.primary {
    background: #475569 !important;
    border-color: #475569 !important;
    color: white !important;
}

button.primary:hover {
    background: #334155 !important;
    border-color: #334155 !important;
}

button.secondary {
    background: #e5e7eb !important;
    color: #374151 !important;
    border-color: #d1d5db !important;
}

button.secondary:hover {
    background: #d1d5db !important;
}

/* Slider accent */
input[type="range"] {
    accent-color: #64748b;
}
"""


# 5. Build a cleaner layout using gr.Blocks
with gr.Blocks(title="The Rudeness Lab") as view:
    gr.HTML(
        """
        <div id="header-banner">
            <h1>The Rudeness Lab</h1>
            <p>Give Gautam a compliment and brace yourself for the response.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            compliment_input = gr.Textbox(
                label="Compliment for Gautam",
                placeholder="Type something nice or ugly here...",
                lines=3,
                elem_classes="styled-box"
            )
            with gr.Row():
                rudeness_slider = gr.Slider(
                    label="Rudeness Level",
                    value=2,
                    minimum=1,
                    maximum=5,
                    step=1
                )
                model_dropdown = gr.Dropdown(
                    choices=[
                        ("Claude Sonnet 5.5", "claude-sonnet-5-5"),
                        ("GPT-6 Luna", "gpt-6-luna"),
                        ("Gemini 3.5 Flash-Lite", "gemini-3.5-flash-lite")
                    ],
                    value="gpt-6-luna",
                    label="Select AI Model"
                )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Deliver Compliment", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="Gautam's Response",
                lines=8,
                interactive=False,
                buttons=["copy"],
                elem_classes="styled-box"
            )

    # Connect buttons to functions
    submit_btn.click(
        fn=message_gpt,
        inputs=[compliment_input, rudeness_slider, model_dropdown],
        outputs=output_box
    )
    clear_btn.click(
        fn=lambda: ("", 1, ""),
        outputs=[compliment_input, rudeness_slider, output_box]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    view.launch(
        server_name="0.0.0.0",
        server_port=port,
        auth=auth_list,
        auth_message="<h2>Be Rude till Doomsday</h2><p>Please sign in to continue.</p>",
        theme=custom_theme,
        css=custom_css
    )
