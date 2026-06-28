from __future__ import annotations

from typing import Optional

import librosa
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base_interface import BaseModel


class MiniCPMClient(BaseModel):
    def __init__(self, model_id: str = "openbmb/MiniCPM-o-2_6", device: str = "cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None

    def load_model(self):
        if self.model is not None:
            return

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="sdpa",
            init_vision=True,
            init_audio=True,
            init_tts=False,
        ).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

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
        audio_np, _ = librosa.load(audio_path, sr=16000, mono=True)

        content = []
        if image_path:
            content.append(Image.open(image_path).convert("RGB"))
        if image_path_2:
            content.append(Image.open(image_path_2).convert("RGB"))
        content.extend([audio_np, full_prompt])

        system_message = self.model.get_sys_prompt(mode="omni", language="en")
        user_message = {"role": "user", "content": content}
        messages = [system_message, user_message]

        response = self.model.chat(
            msgs=messages,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            sampling=True,
            omni_input=True,
            use_tts_template=False,
            generate_audio=False,
        )

        if isinstance(response, dict):
            return response.get("text", "")
        return response if isinstance(response, str) else str(response)
