from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import Optional, List, Union, Dict, Any
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
from qwen_vl_utils import process_vision_info
import uvicorn
import json
from datetime import datetime
import logging
import time
import psutil
import GPUtil
import base64
from PIL import Image
import io
import os
import threading

# Set environment variables to disable compilation cache and avoid CUDA kernel issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # Compatible with A5000

# Model configuration
MODELS = {
    "Qwen2.5-VL-7B-Instruct": {
        "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class": Qwen2_5_VLForConditionalGeneration,
    },
    "Qwen2-VL-7B-Instruct": {
        "path": "Qwen/Qwen2-VL-7B-Instruct",
        "model_class": Qwen2VLForConditionalGeneration,
    },
    "Qwen2-VL-2B-Instruct": {
        "path": "Qwen/Qwen2-VL-2B-Instruct",
        "model_class": Qwen2VLForConditionalGeneration,
    }
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
models = {}
processors = {}
model_locks = {}  # Thread locks for model loading
last_used = {}    # Record last use time of models

# Set default CUDA device
if torch.cuda.is_available():
    # Get GPU information and select the device with maximum memory
    gpus = GPUtil.getGPUs()
    if gpus:
        max_memory_gpu = max(gpus, key=lambda g: g.memoryTotal)
        selected_device = max_memory_gpu.id
        torch.cuda.set_device(selected_device)
        device = torch.device(f"cuda:{selected_device}")
        logger.info(f"Selected GPU {selected_device} ({max_memory_gpu.name}) with {max_memory_gpu.memoryTotal}MB memory")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

class ImageURL(BaseModel):
    url: str

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ['text', 'image_url']:
            raise ValueError(f"Invalid content type: {v}")
        return v

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ['system', 'user', 'assistant']:
            raise ValueError(f"Invalid role: {v}")
        return v

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: Union[str, List[Any]]) -> Union[str, List[MessageContent]]:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return [MessageContent(**item) if isinstance(item, dict) else item for item in v]
        raise ValueError("Content must be either a string or a list of content items")

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    response_format: Optional[Dict[str, str]] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

class ModelCard(BaseModel):
    id: str
    created: int
    owned_by: str
    permission: List[Dict[str, Any]] = []
    root: Optional[str] = None
    parent: Optional[str] = None
    capabilities: Optional[Dict[str, bool]] = None
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

def process_base64_image(base64_string: str) -> Image.Image:
    """Process base64 image data and return PIL Image"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_string:
            base64_string = base64_string.split('base64,')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        return image
    except Exception as e:
        logger.error(f"Error processing base64 image: {str(e)}")
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def log_system_info():
    """Log system resource information"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        gpu_info = []
        if torch.cuda.is_available():
            for gpu in GPUtil.getGPUs():
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': f"{gpu.load*100}%",
                    'memory_used': f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB",
                    'temperature': f"{gpu.temperature}°C"
                })
        logger.info(f"System Info - CPU: {cpu_percent}%, RAM: {memory.percent}%, "
                   f"Available RAM: {memory.available/1024/1024/1024:.1f}GB")
        if gpu_info:
            logger.info(f"GPU Info: {gpu_info}")
    except Exception as e:
        logger.warning(f"Failed to log system info: {str(e)}")

def get_or_initialize_model(model_name: str):
    """Get or initialize a model if not already loaded"""
    global models, processors, model_locks, last_used
    
    if model_name not in MODELS:
        available_models = list(MODELS.keys())
        raise ValueError(f"Unsupported model: {model_name}\nAvailable models: {available_models}")
    
    # Initialize lock for the model (if not already done)
    if model_name not in model_locks:
        model_locks[model_name] = threading.Lock()
    
    with model_locks[model_name]:
        if model_name not in models or model_name not in processors:
            try:
                start_time = time.time()
                logger.info(f"Starting {model_name} initialization...")
                log_system_info()
                
                model_config = MODELS[model_name]
                
                # Configure 8-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                    bnb_4bit_quant_type="nf4",
                )
                
                logger.info(f"Loading {model_name} with 8-bit quantization...")
                model = model_config["model_class"].from_pretrained(
                    model_config["path"],
                    quantization_config=quantization_config,
                    device_map={"": device.index if device.type == "cuda" else "cpu"},
                    local_files_only=False
                ).eval()
                
                processor = AutoProcessor.from_pretrained(
                    model_config["path"],
                    local_files_only=False
                )
                
                models[model_name] = model
                processors[model_name] = processor
                
                end_time = time.time()
                logger.info(f"Model {model_name} initialized in {end_time - start_time:.2f} seconds")
                log_system_info()
                
            except Exception as e:
                logger.error(f"Model initialization error for {model_name}: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to initialize model {model_name}: {str(e)}")
        
        # Update last use time
        last_used[model_name] = time.time()
        
        return models[model_name], processors[model_name]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application initialization...")
    try:
        yield
    finally:
        logger.info("Shutting down application...")
        global models, processors
        for model_name, model in models.items():
            try:
                del model
                logger.info(f"Model {model_name} unloaded")
            except Exception as e:
                logger.error(f"Error during cleanup of {model_name}: {str(e)}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        
        models = {}
        processors = {}
        logger.info("Shutdown complete")

app = FastAPI(
    title="Qwen2.5-VL API",
    description="OpenAI-compatible API for Qwen2.5-VL vision-language model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models"""
    model_cards = []
    for model_name in MODELS.keys():
        model_cards.append(
            ModelCard(
                id=model_name,
                created=1709251200,
                owned_by="Qwen",
                permission=[{
                    "id": f"modelperm-{model_name}",
                    "created": 1709251200,
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }],
                capabilities={
                    "vision": True,
                    "chat": True,
                    "embeddings": False,
                    "text_completion": True
                },
                context_window=4096,
                max_tokens=2048
            )
        )
    return ModelList(data=model_cards)

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests with vision support"""
    try:
        # Get or initialize requested model
        model, processor = get_or_initialize_model(request.model)
        
        request_start_time = time.time()
        logger.info(f"Received chat completion request for model: {request.model}")
        logger.info(f"Request content: {request.model_dump_json()}")
        
        messages = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                processed_content = []
                for content_item in msg.content:
                    if content_item.type == "text":
                        processed_content.append({
                            "type": "text",
                            "text": content_item.text
                        })
                    elif content_item.type == "image_url":
                        if "url" in content_item.image_url:
                            if content_item.image_url["url"].startswith("data:image"):
                                processed_content.append({
                                    "type": "image",
                                    "image": process_base64_image(content_item.image_url["url"])
                                })
                messages.append({"role": msg.role, "content": processed_content})
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Ensure input data is on the correct device
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move all tensors to specified device
        input_tensors = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **input_tensors,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        
        # Get input length and trim generated IDs
        input_length = input_tensors['input_ids'].shape[1]
        generated_ids_trimmed = generated_ids[:, input_length:]
        
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        if request.response_format and request.response_format.get("type") == "json_object":
            try:
                if response.startswith('```'):
                    response = '\n'.join(response.split('\n')[1:-1])
                if response.startswith('json'):
                    response = response[4:].lstrip()
                content = json.loads(response)
                response = json.dumps(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON response: {str(e)}")
        
        total_time = time.time() - request_start_time
        logger.info(f"Request completed in {total_time:.2f} seconds")
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": input_length,
                "completion_tokens": len(generated_ids_trimmed[0]),
                "total_tokens": input_length + len(generated_ids_trimmed[0])
            }
        )
    except Exception as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    log_system_info()
    return {
        "status": "healthy",
        "loaded_models": list(models.keys()),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model_status")
async def model_status():
    """Get the status of all models"""
    status = {}
    for model_name in MODELS:
        status[model_name] = {
            "loaded": model_name in models,
            "last_used": last_used.get(model_name, None),
            "available": model_name in MODELS
        }
    return status

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9192)