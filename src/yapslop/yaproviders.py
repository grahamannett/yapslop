from dataclasses import dataclass

import httpx
import torch
import torchaudio

from yapslop.convo_dto import TextOptions
from yapslop.convo_helpers import MessageType
from yapslop.generator import Segment, load_csm_1b
from yapslop.yap_common import DEVICE


# using _Providers as a registry, can store the type of provider, or use like "ollama" in future
# avoid using too deep a structure for all of this as will make future threading/multiprocessing easier
_Providers = {}


def add_provider(name: str, use_dc: bool = True):
    """
    Decorator to register a provider class in the _Providers registry.

    Args:
        name: Name to register the provider under
        use_dc: Whether to wrap the class in @dataclass
    """

    def wrapper(func):  # move dataclass to this for slightly clearer code
        _Providers[name] = dataclass(func) if use_dc else func
        return func

    return wrapper


def ProvidersSetup(configs: dict[str, dict]):
    """
    Initialize provider instances from a config dictionary.

    Args:
        configs: Dictionary mapping provider names to their config dicts

    Returns:
        List of initialized provider instances
    """
    return [_Providers[key](**config) for key, config in configs.items()]


@add_provider("text")
class TextProvider:
    """Base class for model providers, can switch to using pydantic-ai/vllm/etc later"""

    client: httpx.AsyncClient
    model_name: str = "gemma3:latest"

    def _from_resp(self, resp: httpx.Response) -> dict:
        """Parse and validate response from API"""
        resp.raise_for_status()
        return resp.json()

    async def __call__(self, *args, **kwargs) -> str | dict:
        return await self.chat_oai(*args, **kwargs)

    async def chat_oai(self, messages: MessageType, model_options: TextOptions | dict = {}, **kwargs) -> str:
        """
        Generate text using OpenAI API endpoint.

        Args:
            messages: List of message dicts to send to the API
            model_options: TextOptions instance or dict of model parameters
            **kwargs: Additional args to pass to model_options.asdict()

        Returns:
            Generated text response
        """
        if isinstance(model_options, TextOptions):
            model_options = model_options.asdict(**kwargs)

        response = await self.client.post(
            "/v1/chat/completions",  # openai-compatible endpoint
            json={
                "model": self.model_name,
                "messages": messages,
                **model_options,
            },
        )
        response_data = self._from_resp(response)
        return response_data["choices"][0]["message"]["content"]

    async def chat_ollama(
        self,
        messages: MessageType,
        stream: bool = False,
        tools: list[dict] | None = None,
        model_options: TextOptions | dict | None = None,
        **kwargs,
    ) -> dict:
        """
        Generate a chat completion response using Ollama's API.

        Args:
            messages: List of message dicts to send to the API
            stream: Whether to stream the response
            tools: Optional list of tools to make available
            model_options: TextOptions instance or dict of model parameters
            **kwargs: Additional args to pass to model_options.asdict()

        Returns:
            Response data from the Ollama API
        """

        if isinstance(model_options, TextOptions):
            model_options = model_options.ollama_fmt().asdict(**kwargs)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        if model_options:
            payload["options"] = model_options
        if tools:
            payload["tools"] = tools

        response = await self.client.post("/api/chat", json=payload)
        response_data = self._from_resp(response)
        return response_data


@add_provider("audio")
class AudioProvider:
    repo_id: str = "sesame/csm-1b"
    device: str = DEVICE

    def __post_init__(self):
        self.generator = load_csm_1b(device=self.device)

    @property
    def sample_rate(self) -> int:
        return self.generator.sample_rate

    def generate_audio(
        self,
        text: str,
        speaker_id: int,
        context: list[Segment] = [],
        max_audio_length_ms: int = 90_000,
    ):
        """
        Generate audio for the given text and speaker.

        Args:
            text: The text to convert to speech
            speaker_id: Speaker ID for the CSM model (0, 1, etc.)
            context: Optional list of previous segments for context
            max_audio_length_ms: Maximum audio length in milliseconds

        Returns:
            torch.Tensor containing the generated audio
        """
        audio = self.generator.generate(
            text=text,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
        )

        return audio

    def save_audio(self, audio: torch.Tensor, file_path: str):
        """
        Save the generated audio to a file.

        Args:
            audio: Audio tensor to save
            file_path: Path to save the audio file to
        """
        torchaudio.save(file_path, audio.unsqueeze(0).cpu(), self.sample_rate)
