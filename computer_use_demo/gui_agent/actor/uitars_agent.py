import json
import re
from openai import OpenAI

from computer_use_demo.gui_agent.llm_utils.oai import encode_image
from computer_use_demo.tools.screen_capture import get_screenshot
from computer_use_demo.tools.logger import logger, truncate_string


class UITARS_Actor:
    """
    In OOTB, we use the default grounding system prompt form UI_TARS repo, and then convert its action to our action format.
    """

    _NAV_SYSTEM_GROUNDING = """
You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```Action: ...```

## Action Space
click(start_box='<|box_start|>(x1,y1)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use \"\" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.

## Note
- Do not generate any other text.
"""

    def __init__(self, ui_tars_url, output_callback, api_key="", selected_screen=0):

        self.ui_tars_url = ui_tars_url
        self.ui_tars_client = OpenAI(base_url=self.ui_tars_url, api_key=api_key)
        self.selected_screen = selected_screen
        self.output_callback = output_callback

        self.grounding_system_prompt = self._NAV_SYSTEM_GROUNDING.format()


    def __call__(self, messages):

        task = messages
        
        # take screenshot
        screenshot, screenshot_path = get_screenshot(selected_screen=self.selected_screen, resize=True, target_width=1920, target_height=1080)
        screenshot_path = str(screenshot_path)
        screenshot_base64 = encode_image(screenshot_path)

        logger.info(f"Sending messages to UI-TARS on {self.ui_tars_url}: {task}, screenshot: {screenshot_path}")

        response = self.ui_tars_client.chat.completions.create(
            model="ui-tars",
            messages=[
                {"role": "system", "content": self.grounding_system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": task},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                    ]
                },
                ],
            max_tokens=256,
            temperature=0
            )
        
        ui_tars_action = response.choices[0].message.content
        converted_action = convert_ui_tars_action_to_json(ui_tars_action)
        response = str(converted_action)

        response = {'content': response, 'role': 'assistant'}
        return response



def convert_ui_tars_action_to_json(action_str: str) -> str:
    """
    Converts an action line such as:
      Action: click(start_box='(153,97)')
    into a JSON string of the form:
      {
        "action": "CLICK",
        "value": null,
        "position": [153, 97]
      }
    """
    
    # Strip leading/trailing whitespace and remove "Action: " prefix if present
    action_str = action_str.strip()
    if action_str.startswith("Action:"):
        action_str = action_str[len("Action:"):].strip()

    # Mappings from old action names to the new action schema
    ACTION_MAP = {
        "click": "CLICK",
        "type": "INPUT",
        "scroll": "SCROLL",
        "wait": "STOP",        # TODO: deal with "wait()"
        "finished": "STOP",
        "call_user": "STOP",
        "hotkey": "HOTKEY",    # We break down the actual key below (Enter, Esc, etc.)
    }

    # Prepare a structure for the final JSON
    # Default to no position and null value
    output_dict = {
        "action": None,
        "value": None,
        "position": None
    }

    # 1) CLICK(...) e.g. click(start_box='(153,97)')
    match_click = re.match(r"^click\(start_box='\(?(\d+),\s*(\d+)\)?'\)$", action_str)
    if match_click:
        x, y = match_click.groups()
        output_dict["action"] = ACTION_MAP["click"]
        output_dict["position"] = [int(x), int(y)]
        return json.dumps(output_dict)

    # 2) HOTKEY(...) e.g. hotkey(key='Enter')
    match_hotkey = re.match(r"^hotkey\(key='([^']+)'\)$", action_str)
    if match_hotkey:
        key = match_hotkey.group(1).lower()
        if key == "enter":
            output_dict["action"] = "ENTER"
        elif key == "esc":
            output_dict["action"] = "ESC"
        else:
            # Otherwise treat it as some generic hotkey
            output_dict["action"] = ACTION_MAP["hotkey"]
            output_dict["value"] = key
        return json.dumps(output_dict)

    # 3) TYPE(...) e.g. type(content='some text')
    match_type = re.match(r"^type\(content='([^']*)'\)$", action_str)
    if match_type:
        typed_content = match_type.group(1)
        output_dict["action"] = ACTION_MAP["type"]
        output_dict["value"] = typed_content
        # If you want a position (x,y) you need it in your string. Otherwise it's omitted.
        return json.dumps(output_dict)

    # 4) SCROLL(...) e.g. scroll(start_box='(153,97)', direction='down')
    #    or scroll(start_box='...', direction='down')
    match_scroll = re.match(
        r"^scroll\(start_box='[^']*'\s*,\s*direction='(down|up|left|right)'\)$",
        action_str
    )
    if match_scroll:
        direction = match_scroll.group(1)
        output_dict["action"] = ACTION_MAP["scroll"]
        output_dict["value"] = direction
        return json.dumps(output_dict)

    # 5) WAIT() or FINISHED() or CALL_USER() etc.
    if action_str in ["wait()", "finished()", "call_user()"]:
        base_action = action_str.replace("()", "")
        if base_action in ACTION_MAP:
            output_dict["action"] = ACTION_MAP[base_action]
        else:
            output_dict["action"] = "STOP"
        return json.dumps(output_dict)

    # If none of the above patterns matched, you can decide how to handle
    # unknown or unexpected action lines:
    output_dict["action"] = "STOP"
    return json.dumps(output_dict)