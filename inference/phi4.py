from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from .base_interface import BaseModel


class Phi4Client(BaseModel):
    def __init__(self, model_id: str = "microsoft/Phi-4-multimodal-instruct", device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is not None:
            return

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = self.device if self.device != "cuda" else "auto"

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model.load_adapter(
            self.model_id,
            adapter_name="speech",
            device_map=device_map,
            adapter_kwargs={"subfolder": "speech-lora"},
        )
        self.model.set_adapter("speech")

    def generate(
        self,
        prompt: str,
        audio_path: str,
        image_path: Optional[str] = None,
        image_path_2: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        if self.model is None:
            self.load_model()

        full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
        content = []
        if image_path:
            content.append({"type": "image", "path": image_path})
        if image_path_2:
            content.append({"type": "image", "path": image_path_2})
        content.append({"type": "audio", "path": audio_path})
        content.append({"type": "text", "text": full_prompt})

        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            # Keep decoding stochastic without hard-coding temperature.
            do_sample=True,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
