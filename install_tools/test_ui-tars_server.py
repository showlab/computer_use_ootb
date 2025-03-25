from openai import OpenAI
from computer_use_demo.gui_agent.llm_utils.oai import encode_image

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

def get_prompt_grounding(task):
    return f"""{task}"""

task = """
```json
{{  "Observation": "I am on the google homepage of the Chrome browser.",
    "Thinking": "The user wants to buy a lap-top on Amazon.com, so I need to click on the address (search) bar of Chrome for entering the 'Amazon.com'.",
    "Next Action": ["I need to click DSML"],
    "Expectation": "The search button is activated after being clicked, ready to input."
}}```
"""

task = """
```json
{{  
"Observation": "I am on the google homepage of the Chrome browser.",
"Thinking": "The user wants to click DSML",
"Next Action": ["I need to click DSML"],
}}```
"""

task = """
```json
{{  
"Observation": "I am on the google homepage of the Chrome browser.",
"Thinking": "The user wants to click Youtube",
"Next Action": ["I need to click Youtube"],
}}```
"""

if __name__ == "__main__":

    ui_tars_url = "https://your_api_to_uitars.com/v1"
    ui_tars_client = OpenAI(base_url=ui_tars_url, api_key="")
    grounding_system_prompt = _NAV_SYSTEM_GROUNDING.format()
    screenshot_base64 = encode_image("./chrome.png")
    prompted_message = get_prompt_grounding(task)

    print(f"grounding_system_prompt, {grounding_system_prompt}, \
            prompted_message: {prompted_message}")

    response = ui_tars_client.chat.completions.create(
        model="ui-tars",
        messages=[
            {"role": "user", "content": grounding_system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": prompted_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_base64}"}}
                ]
            },
            ],
        max_tokens=128,
        temperature=0
        )

    ui_tars_action = response.choices[0].message.content

    print(response.choices[0].message.content)


