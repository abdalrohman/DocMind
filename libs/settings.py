from typing import Dict, Optional

from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings


class LLMSettings(BaseSettings):
    """Define general settings for llm"""

    openai_model: str = Field(default="gpt-3.5-turbo-0125")
    gemini_model: str = Field(default="gemini-1.5-pro-latest")
    groq_model: str = Field(default="mixtral-8x7b-32768")
    fireworks_model: str = Field(
        default="accounts/fireworks/models/yi-34b-200k-capybara"
    )

    temperature: float = 0.0
    max_tokens: Optional[int] = Field(default=32768)
    streaming: bool = True
    max_retries: int = 6
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")
    openai_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    openai_proxy: Optional[str] = None

    # openai settings
    top_p: float = 1
    top_k: int = 40
    frequency_penalty: int = 0
    presence_penalty: int = 0
    logit_bias: Optional[Dict[str, float]] = None
    stop: Optional[str] = None

    # gemini settings
    convert_system_message_to_human: bool = True
    safety_settings: Optional[Dict[HarmCategory, HarmBlockThreshold]] = None


llm_settings = LLMSettings()
