from abc import ABC, abstractmethod
from typing import Optional
import asyncio

class BaseModel(ABC):
    @abstractmethod
    def load_model(self):
        """
        Load the model and processor.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, audio_path: str, image_path: Optional[str] = None, image_path_2: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response given the inputs (Synchronous).
        """
        pass

    async def generate_async(self, prompt: str, audio_path: str, image_path: Optional[str] = None, image_path_2: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response given the inputs (Asynchronous).
        Default implementation wraps synchronous generate in a thread.
        Subclasses should override this for true async support (e.g., API calls).
        """
        return await asyncio.to_thread(self.generate, prompt, audio_path, image_path, image_path_2, system_prompt)
