import os
import ast
import base64
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pyautogui
import requests
import torch
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from computer_use_demo.gui_agent.llm_utils.oai import encode_image
from computer_use_demo.tools.colorful_text import colorful_text_showui, colorful_text_vlm
from computer_use_demo.tools.screen_capture import get_screenshot

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ShowUIActor:
    _NAV_SYSTEM = """
    You are an assistant trained to navigate the {_APP} screen. 
    Given a task instruction, a screen observation, and an action history sequence, 
    output the next action and wait for the next observation. 
    Here is the action space:
    {_ACTION_SPACE}
    """

    _NAV_FORMAT = """
    Format the action as a dictionary with the following keys:
    {'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}
    
    If value or position is not applicable, set it as None.
    Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
    Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
    """

    action_map = {
    'desktop': """
        1. CLICK: Click on an element, value is not applicable and the position [x,y] is required. 
        2. INPUT: Type a string into an element, value is a string to type and the position [x,y] is required. 
        3. HOVER: Hover on an element, value is not applicable and the position [x,y] is required.
        4. ENTER: Enter operation, value and position are not applicable.
        5. SCROLL: Scroll the screen, value is the direction to scroll and the position is not applicable.
        6. ESC: ESCAPE operation, value and position are not applicable.
        7. PRESS: Long click on an element, value is not applicable and the position [x,y] is required. 
        """,
    'phone': """
        1. INPUT: Type a string into an element, value is not applicable and the position [x,y] is required. 
        2. SWIPE: Swipe the screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
        3. TAP: Tap on an element, value is not applicable and the position [x,y] is required.
        4. ANSWER: Answer the question, value is the status (e.g., 'task complete') and the position is not applicable.
        5. ENTER: Enter operation, value and position are not applicable.
        """
    }

    def __init__(self, model_path, output_callback, device=torch.device("cpu"), split='desktop', selected_screen=0,
                 max_pixels=1344, awq_4bit=False):
        self.device = device
        self.split = split
        self.selected_screen = selected_screen
        self.output_callback = output_callback
        
        if not model_path or not os.path.exists(model_path) or not os.listdir(model_path):
            if awq_4bit:
                model_path = "showlab/ShowUI-2B-AWQ-4bit"
            else:
                model_path = "showlab/ShowUI-2B"
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        ).to(self.device)
        self.model.eval()
        
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = max_pixels * 28 * 28
        # self.max_pixels = 1344 * 28 * 28
        
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            # "./Qwen2-VL-2B-Instruct",
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )
        self.system_prompt = self._NAV_SYSTEM.format(
            _APP=split,
            _ACTION_SPACE=self.action_map[split]
        )
        self.action_history = ''  # Initialize action history

    def __call__(self, messages):

        task = messages
        
        # screenshot
        screenshot, screenshot_path = get_screenshot(selected_screen=self.selected_screen, resize=True, target_width=1920, target_height=1080)
        screenshot_path = str(screenshot_path)
        image_base64 = encode_image(screenshot_path)
        self.output_callback(f'Screenshot for {colorful_text_showui}:\n<img src="data:image/png;base64,{image_base64}">', sender="bot")

        # Use system prompt, task, and action history to build the messages
        if len(self.action_history) == 0:
            messages_for_processor = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                        {"type": "image", "image": screenshot_path, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                        {"type": "text", "text": f"Task: {task}"}
                    ],
                }
            ]
        else:
            # https://github.com/showlab/ShowUI/issues/5
            messages_for_processor = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                        {"type": "image", "image": screenshot_path, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                        {"type": "text", "text": f"Task: {task}"},
                        {"type": "text", "text": self.action_history},
                    ],
                }
            ]
        
        text = self.processor.apply_chat_template(
            messages_for_processor, tokenize=False, add_generation_prompt=True,
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
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # dummy output test
        # output_text = "{'action': 'CLICK', 'value': None, 'position': [0.49, 0.42]}"

        # Update action history
        self.action_history += output_text + '\n'

        # Return response in expected format
        response = {'content': output_text, 'role': 'assistant'}
        return response


    def parse_showui_output(self, output_text):
        try:
            # Ensure the output is stripped of any extra spaces
            output_text = output_text.strip()

            # Wrap the input in brackets if it looks like a single dictionary
            if output_text.startswith("{") and output_text.endswith("}"):
                output_text = f"[{output_text}]"

            # Validate if the output resembles a list of dictionaries
            if not (output_text.startswith("[") and output_text.endswith("]")):
                raise ValueError("Output does not look like a valid list or dictionary.")

            # Parse the output using ast.literal_eval
            parsed_output = ast.literal_eval(output_text)

            # Ensure the result is a list
            if isinstance(parsed_output, dict):
                parsed_output = [parsed_output]
            elif not isinstance(parsed_output, list):
                raise ValueError("Parsed output is neither a dictionary nor a list.")

            # Ensure all elements in the list are dictionaries
            if not all(isinstance(item, dict) for item in parsed_output):
                raise ValueError("Not all items in the parsed output are dictionaries.")

            return parsed_output

        except Exception as e:
            print(f"Error parsing output: {e}")
            return None
