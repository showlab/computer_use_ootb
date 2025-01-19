import base64
import logging
from .oai import run_oai_interleaved
from .gemini import run_gemini_interleaved

def run_llm(prompt, llm="gpt-4o-mini", max_tokens=256, temperature=0, stop=None):
    log_prompt(prompt)
    
    # turn string prompt into list
    if isinstance(prompt, str):
        prompt = [prompt]
    elif isinstance(prompt, list):
        pass
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")
    
    # Optimize prompt for cost-efficiency
    prompt = optimize_prompt(prompt)
    
    if llm.startswith("gpt"): # gpt series
        out = run_oai_interleaved(
            prompt, 
            llm, 
            max_tokens, 
            temperature, 
            stop
        )
    elif llm.startswith("gemini"): # gemini series
        out = run_gemini_interleaved(
            prompt, 
            llm, 
            max_tokens,
            temperature, 
            stop
        )
    else:
        raise ValueError(f"Invalid llm: {llm}")
    logging.info(
        f"========Output for {llm}=======\n{out}\n============================")
    return out

def log_prompt(prompt):
    prompt_display = [prompt] if isinstance(prompt, str) else prompt
    prompt_display = "\n\n".join(prompt_display)
    logging.info(
        f"========Prompt=======\n{prompt_display}\n============================")

def optimize_prompt(prompt):
    """
    Optimize the prompt to minimize token usage by using concise language and removing unnecessary details.
    """
    optimized_prompt = []
    for p in prompt:
        # Remove unnecessary details and focus on essential information
        p = p.replace("Please", "").replace("kindly", "").replace("could you", "").replace("would you", "")
        # Use abbreviations where appropriate
        p = p.replace("information", "info").replace("application", "app")
        optimized_prompt.append(p.strip())
    return optimized_prompt
