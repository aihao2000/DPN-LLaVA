"""
Live Avatar demo for DPN-LLaVA.

Launches a Gradio interface with webcam (or image upload) input. The model
runs locally — no separate controller / worker processes are required.

Usage:
    python -m llava_hr.serve.live_avatar \
        --model-path AisingioroHao0/dpn-llava-v1.5-7b \
        [--model-base <base-model-path>] \
        [--device cuda] \
        [--load-4bit | --load-8bit] \
        [--conv-mode llava_v1] \
        [--temperature 0.2] \
        [--max-new-tokens 512] \
        [--share]
"""

import argparse
import threading

import gradio as gr
import torch
from PIL import Image
from transformers import TextIteratorStreamer

from llava_hr.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava_hr.conversation import SeparatorStyle, conv_templates
from llava_hr.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava_hr.model.builder import load_pretrained_model
from llava_hr.utils import disable_torch_init

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_image_processor = None
_args = None  # parsed CLI args


def _load_model(args):
    global _tokenizer, _model, _image_processor
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    _tokenizer, _model, _image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
    )


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _pick_conv_mode(model_name: str, override: str | None) -> str:
    if override:
        return override
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


def _run_inference(image: Image.Image, prompt: str, history: list, args) -> str:
    """
    Run a single forward pass and stream tokens; returns the full response.

    *history* is a list of [user_text, assistant_text] pairs (Gradio chatbot
    format).  The image is only embedded in the *first* user turn.
    """
    model_name = get_model_name_from_path(args.model_path)
    conv_mode = _pick_conv_mode(model_name, args.conv_mode)
    conv = conv_templates[conv_mode].copy()

    # Replay history into the conversation (text only for past turns)
    for user_msg, asst_msg in history[:-1]:
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], asst_msg)

    # Build the current user message with the image token
    user_text = history[-1][0]
    if image is not None and len(history) == 1:
        # First turn: prepend image token
        if _model.config.mm_use_im_start_end:
            user_text = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + user_text
            )
        else:
            user_text = DEFAULT_IMAGE_TOKEN + "\n" + user_text

    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    # Tokenise
    input_ids = (
        tokenizer_image_token(
            full_prompt, _tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(_model.device)
    )

    # Prepare image tensor (only needed on the first turn)
    image_tensor = None
    if image is not None and len(history) == 1:
        image_tensor = process_images([image], _image_processor, _model.config)
        if isinstance(image_tensor, list):
            image_tensor = [
                img.to(_model.device, dtype=torch.float16) for img in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(_model.device, dtype=torch.float16)

    # Stop token
    stop_str = (
        conv.sep
        if conv.sep_style != SeparatorStyle.TWO
        else conv.sep2
    )

    streamer = TextIteratorStreamer(
        _tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=30.0
    )

    generate_kwargs = dict(
        inputs=input_ids,
        images=image_tensor,
        do_sample=(args.temperature > 0),
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        streamer=streamer,
        use_cache=True,
    )

    thread = threading.Thread(
        target=_model.generate, kwargs=generate_kwargs
    )
    thread.start()

    generated = ""
    for token_text in streamer:
        generated += token_text
        if generated.endswith(stop_str):
            generated = generated[: -len(stop_str)].rstrip()
            break

    thread.join()
    return generated.strip()


# ---------------------------------------------------------------------------
# Gradio UI callbacks
# ---------------------------------------------------------------------------

def user_submit(user_message: str, image, history: list):
    """Append the user message (with image thumbnail) to history."""
    if not user_message.strip() and image is None:
        return history, history, ""
    history = history + [[user_message, None]]
    return history, history, ""


def bot_respond(image, history: list):
    """Generate the assistant reply and stream it into the chatbot."""
    if not history:
        yield history
        return

    pil_image = None
    if image is not None:
        if not isinstance(image, Image.Image):
            pil_image = Image.fromarray(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

    response = _run_inference(pil_image, history[-1][0], history, _args)
    history[-1][1] = response
    yield history


def clear_all():
    return [], [], None, ""


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

_TITLE = """
# 🌋 DPN-LLaVA Live Avatar
**Dynamic Pyramid Network** — real-time visual conversation assistant.

Upload an image **or** capture one from your webcam, then chat with the model.
"""


def build_demo():
    with gr.Blocks(title="DPN-LLaVA Live Avatar") as demo:
        gr.Markdown(_TITLE)

        with gr.Row():
            # ---- Left panel: image input ----
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Image / Webcam",
                    source="webcam",
                    type="pil",
                    mirror_webcam=False,
                    tool="editor",
                )
                gr.Markdown(
                    "_Switch between **Upload** and **Webcam** using the icons "
                    "in the image widget toolbar._"
                )
                with gr.Accordion("Generation parameters", open=False):
                    temperature_slider = gr.Slider(
                        0.0, 1.0, value=_args.temperature, step=0.05,
                        label="Temperature",
                    )
                    max_tokens_slider = gr.Slider(
                        64, 2048, value=_args.max_new_tokens, step=64,
                        label="Max new tokens",
                    )

            # ---- Right panel: chat ----
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                with gr.Row():
                    msg_box = gr.Textbox(
                        show_label=False,
                        placeholder="Ask something about the image…",
                        container=False,
                        scale=8,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️  Clear conversation")

        # Internal state: Gradio chatbot history
        state = gr.State([])

        # Wire up generation-parameter sliders to _args (live update)
        def update_temperature(val):
            _args.temperature = val

        def update_max_tokens(val):
            _args.max_new_tokens = int(val)

        temperature_slider.change(update_temperature, inputs=temperature_slider)
        max_tokens_slider.change(update_max_tokens, inputs=max_tokens_slider)

        # Submit flow: user turn → bot turn
        submit_event = msg_box.submit(
            user_submit,
            inputs=[msg_box, image_input, state],
            outputs=[chatbot, state, msg_box],
        ).then(
            bot_respond,
            inputs=[image_input, state],
            outputs=[chatbot],
        ).then(
            lambda h: h,
            inputs=[chatbot],
            outputs=[state],
        )

        send_btn.click(
            user_submit,
            inputs=[msg_box, image_input, state],
            outputs=[chatbot, state, msg_box],
        ).then(
            bot_respond,
            inputs=[image_input, state],
            outputs=[chatbot],
        ).then(
            lambda h: h,
            inputs=[chatbot],
            outputs=[state],
        )

        clear_btn.click(
            clear_all,
            outputs=[chatbot, state, image_input, msg_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DPN-LLaVA Live Avatar demo")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path or HuggingFace repo of the fine-tuned DPN-LLaVA model")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base LLM path (only needed for LoRA / projector-only checkpoints)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"])
    parser.add_argument("--conv-mode", type=str, default=None,
                        help="Conversation template (auto-detected if omitted)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    print("Loading model …")
    _load_model(_args)
    print("Model loaded. Launching Gradio demo …")
    demo = build_demo()
    demo.queue().launch(
        server_name=_args.host,
        server_port=_args.port,
        share=_args.share,
    )
