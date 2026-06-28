import torch
import logging
from typing import Optional
from .base_interface import BaseModel

class InteractiveOmniClient(BaseModel):
    def __init__(self, device="cuda"):
        self.device = device
        
        # Try to set the default cuda device to the requested one.
        # This helps if internal components use default 'cuda' (which implies index 0)
        if "cuda" in device and ":" in device:
            try:
                # device string format "cuda:X"
                idx = int(device.split(":")[-1])
                torch.cuda.set_device(idx)
            except Exception as e:
                logging.warning(f"Could not set default cuda device to {device}: {e}")

        self.model = None
        self.tokenizer = None
        self.model_path = "sensenova/InteractiveOmni-8B"

    def load_model(self):
        # Lazy import to avoid top-level failures if dependencies are missing
        from transformers import AutoModel, AutoTokenizer
        
        logging.info(f"Loading InteractiveOmni model from {self.model_path}...")
        try:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).eval()

            # InteractiveOmni loads some components (like SigLip) that might default to cuda:0.
            # We must ensure EVERYTHING is on the target device.
            # Using device_map="auto" in from_pretrained could help, but manual .to() is safer for single-device.
            self.model.to(self.device)
            
            # Additional check: iterate over all submodules and force them to device
            # This handles edge cases where some buffers/parameters are missed by top-level .to()
            for param in self.model.parameters():
                param.data = param.data.to(self.device)
            for buffer in self.model.buffers():
                buffer.data = buffer.data.to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True, 
                use_fast=True
            )
            logging.info("InteractiveOmni model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load InteractiveOmni model: {e}")
            raise

    def generate(self, prompt: str, audio_path: str, image_path: Optional[str] = None, image_path_2: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        if self.model is None:
            self.load_model()

        generation_config = dict(max_new_tokens=1024, do_sample=True)
        max_num = 12

        messages = []

        if system_prompt:
             messages.append({
                "role": "system",
                "content": system_prompt
            })

        user_content = []

        # Add image 1
        if image_path:
            user_content.append({
                "type": "image",
                "image": image_path
            })

        # Add image 2
        if image_path_2:
            user_content.append({
                "type": "image",
                "image": image_path_2
            })

        # Add audio (required)
        if audio_path:
            user_content.append({
                "type": "audio",
                "audio": audio_path
            })

        # Add text
        user_content.append({
            "type": "text",
            "text": prompt
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        try:
            # Check if image is present to decide on calling convention (max_num)
            # The error 'unexpected keyword argument max_num' suggests that the model's chat method signature 
            # might have changed or we are using it incorrectly. The README example showed passing max_num 
            # as a positional argument (4th arg) for image-text/video, but keywords might not be supported.
            # Let's try passing it as a positional argument if image_path is present, or omit if not needed.
            
            # Based on README:
            # response = model.chat(tokenizer, generation_config, messages, max_num)
            
            if image_path or image_path_2:
                # Pass max_num as a positional argument
                # Ensure generation_config is not causing device issues (it's a dict, so safe)
                response = self.model.chat(self.tokenizer, generation_config, messages, max_num)
            else:
                response = self.model.chat(self.tokenizer, generation_config, messages)
            
            # Ensure response is a string
            if isinstance(response, tuple):
                 return response[0]
            return str(response)

        except Exception as e:
            logging.error(f"Error in InteractiveOmni generation: {e}")
            return ""
