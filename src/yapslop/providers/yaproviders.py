from dataclasses import dataclass
from typing import Callable, Type


import httpx
import torch
import torchaudio

from yapslop.convo_dto import TextOptions
from yapslop.convo_helpers import MessageType
from yapslop.generator import Segment, load_csm_1b
from yapslop.yap_common import DEVICE
from yapslop.providers.parsers import ReasoningParser, _REQUIRE_PARSERS

# using _Providers as a registry, can store the type of provider, or use like "ollama" in future
# avoid using too deep a structure for all of this as will make future threading/multiprocessing easier
_Providers = {}


def add_provider(name: str, use_dc: bool = True) -> Callable[[Type], Type]:
    """
    Decorator to register a provider class in the _Providers registry.

    Args:
        name: Name to register the provider under
        use_dc: Whether to wrap the class in @dataclass
    """

    def wrapper(cls):  # move dataclass to this for slightly clearer code
        _Providers[name] = dataclass(cls) if use_dc else cls
        return cls

    return wrapper


def AddProvider(cls, name: str):
    # using this is better for IDE to show info about the type, compared to:
    # > @add_provider(type)
    # i'm guessing there is a way to do that the type annotation that is needed for
    # add_provider is not clear to me
    _Providers[name] = cls


def ProvidersSetup(configs: dict[str, dict]):
    """
    Initialize provider instances from a config dictionary.

    Args:
        configs: Dictionary mapping provider names to their config dicts

    Returns:
        List of initialized provider instances
    """
    return [_Providers[key](**config) for key, config in configs.items()]


@dataclass
class TextProvider:
    """Base class for model providers, can switch to using pydantic-ai/vllm/etc later"""

    client: httpx.AsyncClient
    model_name: str = "gemma3:latest"

    _parser: bool | ReasoningParser = False

    def __post_init__(self) -> None:
        # possibly split model_name and check just for name:version
        if self.model_name in _REQUIRE_PARSERS:
            self._parser = ReasoningParser()

    def _from_resp(self, resp: httpx.Response) -> dict:
        """Parse and validate response from API"""
        resp.raise_for_status()
        return resp.json()

    async def __call__(self, *args, **kwargs) -> str | dict:
        return await self.chat_oai(*args, **kwargs)

    async def chat_oai(
        self,
        messages: MessageType,
        model_options: TextOptions | dict = {},
        _get: Callable[[dict], str] = lambda x: x["choices"][0]["message"]["content"],
        **kwargs,
    ) -> str:
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

        raw_resp = await self.client.post(
            "/v1/chat/completions",  # openai-compatible endpoint
            json={
                "model": self.model_name,
                "messages": messages,
                **model_options,
            },
        )
        resp_json = self._from_resp(raw_resp)
        resp = _get(resp_json)
        return resp

    async def chat_ollama(
        self,
        messages: MessageType,
        stream: bool = False,
        tools: list[dict] | None = None,
        model_options: TextOptions | dict | None = None,
        _get: Callable[[dict], str] = lambda x: x["message"]["content"],
        **kwargs,
    ) -> str:
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

        raw_resp = await self.client.post(
            "/api/chat",
            json=payload,
        )
        resp_json = self._from_resp(raw_resp)
        resp = _get(resp_json)
        return resp


@dataclass
class AudioProvider:
    repo_id: str = "sesame/csm-1b"
    device: str = DEVICE

    def __post_init__(self):
        self.generator = load_csm_1b(device=self.device)

    @property
    def sample_rate(self) -> int:
        return self.generator.sample_rate

    async def generate_audio(
        self,
        text: str,
        speaker_id: int,
        context: list[Segment] = [],
        max_audio_length_ms: int = 90_000,
    ) -> torch.Tensor:
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

    def sync_generate_audio(
        self,
        text: str,
        speaker_id: int,
        context: list[Segment] = [],
        max_audio_length_ms: int = 90_000,
    ) -> torch.Tensor:
        """
        Even though generate is sync atm, it will be async if moved to a different process/external service
        """
        return self.generator.generate(
            text=text,
            speaker=speaker_id,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
        )

    def save_audio(self, audio: torch.Tensor, file_path: str):
        """
        Save the generated audio to a file.

        Args:
            audio: Audio tensor to save
            file_path: Path to save the audio file to
        """
        torchaudio.save(file_path, audio.unsqueeze(0).cpu(), self.sample_rate)


AddProvider(TextProvider, "text")
AddProvider(AudioProvider, "audio")
