import os
import logging
import base64
import requests

import dashscope

def is_image_path(text):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif")
    return text.endswith(image_extensions)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_qwen(messages: list, system: str, llm: str, api_key: str, max_tokens=256, temperature=0):
    api_key = api_key or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise ValueError("QWEN_API_KEY is not set")
    
    dashscope.api_key = api_key

    final_messages = [{"role": "system", "content": [{"text": system}]}]
    if type(messages) == list:
        for item in messages:
            contents = []
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            content = [{"image": cnt}]
                        else:
                            content = {"text": cnt}
                    contents.append(content)
                message = {"role": item["role"], "content": contents}
            else:
                contents.append({"text": item})
                message = {"role": "user", "content": contents}
            final_messages.append(message)

    print("[qwen-vl] sending messages:", final_messages)

    response = dashscope.MultiModalConversation.call(
        model='qwen-vl-max-latest',
        messages=final_messages
    )

    try:
        text = response.output.choices[0].message.content[0]['text']
        usage = response.usage
        
        if "total_tokens" not in usage:
            token_usage = int(usage["input_tokens"] + usage["output_tokens"])
        else:
            token_usage = int(usage["total_tokens"])
        
        return text, token_usage
    except Exception as e:
        print(f"Error in interleaved openAI: {e}. This may due to your invalid OPENAI_API_KEY. Please check the response: {response.json()} ")
        return response.json()

if __name__ == "__main__":
    api_key = os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise ValueError("QWEN_API_KEY is not set")
    
    dashscope.api_key = api_key
    
    final_messages = [{"role": "user",
                       "content": [
                           {"text": "What is in the screenshot?"},
                           {"image": "./tmp/outputs/screenshot_0b04acbb783d4706bc93873d17ba8c05.png"}
                           ]
                       }
                    ]
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max-0809', messages=final_messages)
    
    print(response)
    
    text = response.output.choices[0].message.content[0]['text']
    usage = response.usage
    
    if "total_tokens" not in usage:
        if "image_tokens" in usage:
            token_usage = usage["input_tokens"] + usage["output_tokens"] + usage["image_tokens"]
        else:
            token_usage = usage["input_tokens"] + usage["output_tokens"]
    else:
        token_usage = usage["total_tokens"]
    
    print(text, token_usage)
