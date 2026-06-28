import torch
import logging
from typing import Optional
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from .base_interface import BaseModel
import os

class Gemma3nClient(BaseModel):
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        logging.info(f"Loading Gemma 3n model: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()

    def generate(self, prompt: str, audio_path: str, image_path: Optional[str] = None, image_path_2: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        if self.model is None:
            self.load_model()

        messages = []
        if system_prompt:
             messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        user_content = []

        # Image handling (Avatar A and Avatar B)
        if image_path:
            user_content.append({"type": "image", "image": image_path})

        if image_path_2:
            user_content.append({"type": "image", "image": image_path_2})

        # Audio handling (required)
        # Gemma 3n supports audio inputs. We assume the processor handles file paths in 'audio' field.
        # If the model expects specific formatting, the processor's chat template should handle it.
        if audio_path:
             user_content.append({"type": "audio", "audio": audio_path})

        user_content.append({"type": "text", "text": prompt})

        messages.append({
            "role": "user",
            "content": user_content
        })

        # Generate
        with torch.inference_mode():
             # apply_chat_template with tokenize=True returns the model inputs
             inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
             ).to(self.model.device)
             
             input_len = inputs["input_ids"].shape[-1]

             generation = self.model.generate(
                **inputs,
                max_new_tokens=512,
                # Keep decoding stochastic without hard-coding temperature.
                do_sample=True,
             )
             
             # Slice the output to get only the generated tokens
             generation = generation[0][input_len:]
             
             decoded = self.processor.decode(generation, skip_special_tokens=True)
             return decoded
