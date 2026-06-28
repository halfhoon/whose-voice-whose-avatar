import asyncio
import os
from pathlib import Path
from google import genai
from google.genai import types
from .base_interface import BaseModel
from typing import Optional


def mime_type_for_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    return "application/octet-stream"


class GeminiClient(BaseModel):
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running Gemini models.")

        self.client = genai.Client(api_key=self.api_key)

    def load_model(self):
        pass

    async def generate_async(self, prompt: str, audio_path: str, image_path: Optional[str] = None, image_path_2: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """
        Asynchronously calls Gemini API.
        """
        try:
            # 1. Prepare Content Parts
            contents_parts = []

            # Image 1
            if image_path:
                with open(image_path, "rb") as f:
                    img_bytes = f.read()
                contents_parts.append(types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=mime_type_for_path(image_path)
                ))

            # Image 2
            if image_path_2:
                with open(image_path_2, "rb") as f:
                    img_bytes_2 = f.read()
                contents_parts.append(types.Part.from_bytes(
                    data=img_bytes_2,
                    mime_type=mime_type_for_path(image_path_2)
                ))

            # Audio
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            contents_parts.append(types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type_for_path(audio_path)
            ))

            # User Prompt
            contents_parts.append(prompt)

            # 2. Config with System Prompt (use default temperature)
            config = None
            if system_prompt:
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt
                )
            else:
                config = types.GenerateContentConfig()

            # 3. Call API (Async)
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents_parts,
                config=config
            )

            # 4. Extract Text
            texts = []
            if response.candidates:
                for cand in response.candidates:
                    if cand.content and cand.content.parts:
                        for part in cand.content.parts:
                            if hasattr(part, "text") and part.text:
                                texts.append(part.text)
            
            return "\n".join(texts).strip()

        except Exception as e:
            # Error handling logic handled in experiment loop, but returning generic error here
            # to avoid crashing the async gather if exceptions leak.
            # Ideally, throw and catch in the loop, but here we return string error for safety.
            # We will raise it so the retry logic in main loop can catch it.
            raise e

    def generate(self, prompt: str, audio_path: str, image_path: Optional[str] = None, image_path_2: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """
        Synchronous wrapper for async generation.
        """
        return asyncio.run(self.generate_async(prompt, audio_path, image_path, image_path_2, system_prompt))

if __name__ == "__main__":
    print("GeminiClient ready.")
