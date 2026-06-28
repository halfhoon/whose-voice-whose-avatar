import torch
from typing import Optional
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from .base_interface import BaseModel

class Qwen25OmniClient(BaseModel):
    def __init__(self, model_id: str = "Qwen/Qwen2.5-Omni-7B", device: str = "cuda"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = device if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.model is None:
            # device_map="auto" usually overrides .to(device), but we can try to hint or set explicitly if not using auto.
            # Ideally, if user specifies "cuda:1", we should use that.
            # But from_pretrained with device_map="auto" is tricky with explicit device.
            # If explicit device is given (e.g. cuda:1), let's try to pass it to device_map if it's a single device str.
            
            dm = self.device
            if self.device == "cuda":
                dm = "auto"
                
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map=dm,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                enable_audio_output=False, # We only need text output for evaluation
            )
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)

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

        conversation_content = []
        
        # Image first if exists
        if image_path:
            conversation_content.append({"type": "image", "image": image_path})

        if image_path_2:
            conversation_content.append({"type": "image", "image": image_path_2})
        
        # Audio
        conversation_content.append({"type": "audio", "audio": audio_path})
        
        # Text Prompt
        conversation_content.append({"type": "text", "text": prompt})

        # Define default system prompt if none provided
        default_system = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text."
        
        final_system_prompt = system_prompt if system_prompt else default_system

        conversations = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": final_system_prompt
                    }
                ],
            },
            {
                "role": "user",
                "content": conversation_content,
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        with torch.no_grad():
            text_ids, _ = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                # Enable sampling but leave temperature/top-p at model defaults.
                talker_do_sample=True,
            )

        decoded_text = self.processor.batch_decode(text_ids, skip_special_tokens=True)[0]
        return decoded_text

if __name__ == "__main__":
    print("Qwen25OmniClient class defined.")
