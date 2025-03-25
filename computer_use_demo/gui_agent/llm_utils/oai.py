import os
import logging
import base64
import requests
from computer_use_demo.gui_agent.llm_utils.llm_utils import is_image_path, encode_image



def run_oai_interleaved(messages: list, system: str, llm: str, api_key: str, max_tokens=256, temperature=0):

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}

    final_messages = [{"role": "system", "content": system}]

    # image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    if type(messages) == list:
        for item in messages:
            print(f"item: {item}")
            contents = []
            if isinstance(item, dict):
                for cnt in item["content"]:
                    if isinstance(cnt, str):
                        if is_image_path(cnt):
                            base64_image = encode_image(cnt)
                            content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        else:
                            content = {"type": "text", "text": cnt}
                    
                    # if isinstance(cnt, list):
                        
                    contents.append(content)
                message = {"role": item["role"], "content": contents}
                
            elif isinstance(item, str):
                if is_image_path(item):
                    base64_image = encode_image(item)
                    contents.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
                    message = {"role": "user", "content": contents}
                else:
                    contents.append({"type": "text", "text": item})
                    message = {"role": "user", "content": contents}
                    
            else:  # str
                contents.append({"type": "text", "text": item})
                message = {"role": "user", "content": contents}
            
            final_messages.append(message)

    
    elif isinstance(messages, str):
        final_messages.append({"role": "user", "content": messages})

    print("[oai] sending messages:", [f"{k}: {v}, {k}" for k, v in final_messages])

    payload = {
        "model": llm,
        "messages": final_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # "stop": stop,
    }

    # from IPython.core.debugger import Pdb; Pdb().set_trace()

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    try:
        text = response.json()['choices'][0]['message']['content']
        token_usage = int(response.json()['usage']['total_tokens'])
        return text, token_usage
        
    # return error message if the response is not successful
    except Exception as e:
        print(f"Error in interleaved openAI: {e}. This may due to your invalid OPENAI_API_KEY. Please check the response: {response.json()} ")
        return response.json()

def run_ssh_llm_interleaved(messages: list, system: str, llm: str, ssh_host: str, ssh_port: int, max_tokens=256, temperature=0.7, do_sample=True):
    """Send chat completion request to SSH remote server"""
    from PIL import Image
    from io import BytesIO
    def encode_image(image_path: str, max_size=1024) -> str:
        """Convert image to base64 encoding with preprocessing"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB format
                img = img.convert('RGB')
                
                # Scale down if image is too large
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.LANCZOS)
                
                # Convert processed image to base64
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return img_str
        except Exception as e:
            print(f"Image processing failed: {str(e)}")
            raise


    try:
        # Verify SSH connection info
        if not ssh_host or not ssh_port:
            raise ValueError("SSH_HOST and SSH_PORT are not set")
        
        # Build API URL
        api_url = f"http://{ssh_host}:{ssh_port}"
        
        # Prepare message list
        final_messages = []
        
        # Add system message
        if system:
            final_messages.append({
                "role": "system",
                "content": system
            })
            
        # Process user messages
        if type(messages) == list:
            for item in messages:
                contents = []
                if isinstance(item, dict):
                    for cnt in item["content"]:
                        if isinstance(cnt, str):
                            if is_image_path(cnt):
                                base64_image = encode_image(cnt)
                                content = {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            else:
                                content = {
                                    "type": "text",
                                    "text": cnt
                                }
                        contents.append(content)
                    message = {"role": item["role"], "content": contents}
                else:  # str
                    contents.append({"type": "text", "text": item})
                    message = {"role": "user", "content": contents}
                final_messages.append(message)
        elif isinstance(messages, str):
            final_messages.append({
                "role": "user",
                "content": messages
            })

        # Prepare request data
        data = {
            "model": llm,
            "messages": final_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "do_sample": do_sample
        }
        
        print(f"[ssh] Sending chat completion request to model: {llm}")
        print(f"[ssh] sending messages:", final_messages)
        
        # Send request
        response = requests.post(
            f"{api_url}/v1/chat/completions",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        result = response.json()
        
        if response.status_code == 200:
            content = result['choices'][0]['message']['content']
            token_usage = int(result['usage']['total_tokens'])
            print(f"[ssh] Generation successful: {content}")
            return content, token_usage
        else:
            print(f"[ssh] Request failed: {result}")
            raise Exception(f"API request failed: {result}")
            
    except Exception as e:
        print(f"[ssh] Chat completion request failed: {str(e)}")
        raise



if __name__ == "__main__":
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    
    # text, token_usage = run_oai_interleaved(
    #     messages= [{"content": [
    #                     "What is in the screenshot?",   
    #                     "./tmp/outputs/screenshot_0b04acbb783d4706bc93873d17ba8c05.png"],
    #                 "role": "user"
    #                 }],
    #     llm="gpt-4o-mini",
    #     system="You are a helpful assistant",
    #     api_key=api_key,
    #     max_tokens=256,
    #     temperature=0)
    
    # print(text, token_usage)
    text, token_usage = run_ssh_llm_interleaved(
        messages= [{"content": [
                        "What is in the screenshot?",   
                        "tmp/outputs/screenshot_5a26d36c59e84272ab58c1b34493d40d.png"],
                    "role": "user"
                    }],
        llm="Qwen2.5-VL-7B-Instruct",
        ssh_host="10.245.92.68",
        ssh_port=9192,
        max_tokens=256,
        temperature=0.7
    )
    print(text, token_usage)
    # There is an introduction describing the Calyx... 36986
