import time
import os
from typing import Any
import json

from groq import Groq

from our_types import MessageList, SamplerBase

class GroqChatCompletionSampler(SamplerBase):
    """
    Sample from Groq's chat completion API with rate limit handling
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.5,
        max_tokens: int = 7999,
    ):
        self.api_key_name = "GROQ_API_KEY"
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:
                    retry_after = self._handle_rate_limit(e)
                    print(f"Rate limit reached. Waiting for {retry_after} seconds before retrying.")
                    time.sleep(retry_after)
                else:
                    print(f"Unexpected error: {e}")
                    time.sleep(1)  # Add a small delay before retrying

    def _handle_rate_limit(self, exception):
        try:
            error_message = json.loads(exception.message)['error']['message']
            retry_after = float(error_message.split('Please try again in ')[1].split('s.')[0])
        except (KeyError, IndexError, ValueError):
            retry_after = 60  # Default to 60 seconds if we can't parse the message
        return max(retry_after, 1)  # Ensure we wait at least 1 second
