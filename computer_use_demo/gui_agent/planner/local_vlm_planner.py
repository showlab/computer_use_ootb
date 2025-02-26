import json
import asyncio
import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast, Dict, Callable

from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex, APIResponse
from anthropic.types import TextBlock, ToolResultBlockParam
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaMessageParam

from computer_use_demo.tools.screen_capture import get_screenshot
from computer_use_demo.gui_agent.llm_utils.llm_utils import extract_data, encode_image
from computer_use_demo.tools.colorful_text import colorful_text_showui, colorful_text_vlm

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilizing a Windows system with internet access.
* The current date is {datetime.today().strftime('%A, %B %d, %Y')}.
</SYSTEM_CAPABILITY>
"""

MODEL_TO_HF_PATH = {
    "qwen-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2-vl-2b-instruct": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2.5-vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
}


class LocalVLMPlanner:
    def __init__(
        self,
        model: str, 
        provider: str, 
        system_prompt_suffix: str, 
        output_callback: Callable, 
        api_response_callback: Callable,
        max_tokens: int = 4096,
        only_n_most_recent_images: int | None = None,
        selected_screen: int = 0,
        print_usage: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 1344 * 28 * 28
        self.model_name = model
        if model in MODEL_TO_HF_PATH:
            self.hf_path = MODEL_TO_HF_PATH[model]
        else:
            raise ValueError(f"Model {model} not supported for local VLM planner")
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.hf_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.hf_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )
        
        self.provider = provider
        self.system_prompt_suffix = system_prompt_suffix
        self.api_response_callback = api_response_callback
        self.max_tokens = max_tokens
        self.only_n_most_recent_images = only_n_most_recent_images
        self.selected_screen = selected_screen
        self.output_callback = output_callback
        self.system_prompt = self._get_system_prompt() + self.system_prompt_suffix

        self.print_usage = print_usage
        self.total_token_usage = 0
        self.total_cost = 0

           
    def __call__(self, messages: list):
        
        # drop looping actions msg, byte image etc
        planner_messages = _message_filter_callback(messages)  
        print(f"filtered_messages: {planner_messages}")

        # Take a screenshot
        screenshot, screenshot_path = get_screenshot(selected_screen=self.selected_screen)
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)
        self.output_callback(f'Screenshot for {colorful_text_vlm}:\n<img src="data:image/png;base64,{image_base64}">',
                             sender="bot")
        
        if isinstance(planner_messages[-1], dict):
            if not isinstance(planner_messages[-1]["content"], list):
                planner_messages[-1]["content"] = [planner_messages[-1]["content"]]
            planner_messages[-1]["content"].append(screenshot_path)

        print(f"Sending messages to VLMPlanner: {planner_messages}")

        messages_for_processor = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}]
            },
            {
                "role": "user",
                "content": [
                {"type": "image", "image": screenshot_path, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                {"type": "text", "text": f"Task: {''.join(planner_messages)}"}
            ],
        }]
        
        text = self.processor.apply_chat_template(
            messages_for_processor, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages_for_processor)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        vlm_response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"VLMPlanner response: {vlm_response}")
        
        vlm_response_json = extract_data(vlm_response, "json")

        # vlm_plan_str = '\n'.join([f'{key}: {value}' for key, value in json.loads(response).items()])
        vlm_plan_str = ""
        for key, value in json.loads(vlm_response_json).items():
            if key == "Thinking":
                vlm_plan_str += f'{value}'
            else:
                vlm_plan_str += f'\n{key}: {value}'
        
        self.output_callback(f"{colorful_text_vlm}:\n{vlm_plan_str}", sender="bot")
        
        return vlm_response_json


    def _api_response_callback(self, response: APIResponse):
        self.api_response_callback(response)
        

    def reformat_messages(self, messages: list):
        pass

    def _get_system_prompt(self):
        os_name = platform.system()
        return f"""
You are using an {os_name} device.
You are able to use a mouse and keyboard to interact with the computer based on the given task and screenshot.
You can only interact with the desktop GUI (no terminal or application menu access).

You may be given some history plan and actions, this is the response from the previous loop.
You should carefully consider your plan base on the task, screenshot, and history actions.

Your available "Next Action" only include:
- ENTER: Press an enter key.
- ESCAPE: Press an ESCAPE key.
- INPUT: Input a string of text.
- CLICK: Describe the ui element to be clicked.
- HOVER: Describe the ui element to be hovered.
- SCROLL: Scroll the screen, you must specify up or down.
- PRESS: Describe the ui element to be pressed.

Output format:
```json
{{
    "Thinking": str, # describe your thoughts on how to achieve the task, choose one action from available actions at a time.
    "Next Action": "action_type, action description" | "None" # one action at a time, describe it in short and precisely. 
}}
```

One Example:
```json
{{  
    "Thinking": "I need to search and navigate to amazon.com.",
    "Next Action": "CLICK 'Search Google or type a URL'."
}}
```

IMPORTANT NOTES:
1. Carefully observe the screenshot to understand the current state and read history actions.
2. You should only give a single action at a time. for example, INPUT text, and ENTER can't be in one Next Action.
3. Attach the text to Next Action, if there is text or any description for the button. 
4. You should not include other actions, such as keyboard shortcuts.
5. When the task is completed, you should say "Next Action": "None" in the json field.
""" 

def _message_filter_callback(messages):
    filtered_list = []
    try:
        for msg in messages:
            if msg.get('role') in ['user']:
                if not isinstance(msg["content"], list):
                    msg["content"] = [msg["content"]]
                if isinstance(msg["content"][0], TextBlock):
                    filtered_list.append(str(msg["content"][0].text))  # User message
                elif isinstance(msg["content"][0], str):
                    filtered_list.append(msg["content"][0])  # User message
                else:
                    print("[_message_filter_callback]: drop message", msg)
                    continue                
                
            else:
                print("[_message_filter_callback]: drop message", msg)
                continue
            
    except Exception as e:
        print("[_message_filter_callback]: error", e)
                
    return filtered_list