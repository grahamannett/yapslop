from dataclasses import asdict, dataclass, field
from typing import ClassVar

import torch
import torchaudio

from yapslop.generator import Segment


def load_audio(audio_path: str, new_freq: int = 24_000) -> torch.Tensor:
    """
    Load an audio file and resample it to the specified frequency.

    Args:
        audio_path: Path to the audio file to load
        new_freq: Target sample rate to resample to (default: 24kHz)

    Returns:
        Resampled audio tensor
    """
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    return torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=new_freq
    )


@dataclass
class Speaker:
    """Participant in a conversation."""

    name: str
    description: str = ""
    # Make speaker_id a proper field with a default_factory to get the next ID
    speaker_id: int = field(default_factory=lambda: Speaker._get_next_id())

    _next_id: ClassVar[int] = 0

    @classmethod
    def _get_next_id(cls) -> int:
        """
        Generate the next sequential speaker ID.

        Returns:
            The next available speaker ID
        """
        cls._next_id += 1
        return cls._next_id - 1

    def __str__(self) -> str:
        return f"{self.name}"

    def asdict(self) -> dict:
        """
        Convert the Speaker object to a dictionary.

        Returns:
            Dictionary representation of the Speaker
        """
        return asdict(self)


@dataclass
class ConvoTurn:
    """
    Represents a single turn in a conversation.
    """

    speaker: Speaker
    text: str | None = None
    # Add audio field to store generated audio
    audio: torch.Tensor | None = None
    audio_path: str | None = None
    turn_idx: int = 0

    def __str__(self) -> str:
        return f"{self.speaker.name}: {self.text}"

    def to_segment(self) -> Segment:
        """
        Convert this conversation turn to a CSM Segment for use as context.

        Loads audio from file if not already in memory.

        Returns:
            A Segment object containing speaker ID, text, and audio

        Raises:
            ValueError: If neither audio nor audio_path is available
        """
        audio = self.audio
        if audio is None and self.audio_path:
            audio = load_audio(self.audio_path)

        if (audio is None) and (self.audio_path is None):
            raise ValueError("Cannot convert to CSM Segment without audio")

        return Segment(speaker=self.speaker.speaker_id, text=self.text, audio=audio)


@dataclass
class TextOptions:
    max_tokens: int  # ~300?
    temperature: float = 0.8

    def asdict(self, **kwargs) -> dict:
        """
        Convert the TextOptions to a dictionary with optional additional parameters.

        Args:
            **kwargs: Additional key-value pairs to include in the output dictionary

        Returns:
            Dictionary representation of TextOptions merged with additional parameters
        """
        return asdict(self) | kwargs

    # some of the ollama options are different that the openai options, i know ill have to deal with both of these later
    def oai_fmt(self):
        """
        Format options for OpenAI-compatible APIs.

        Returns:
            TextOptions formatted for OpenAI API
        """
        return self

    def ollama_fmt(self):
        """
        Format options for Ollama API.

        Returns:
            TextOptions formatted for Ollama API
        """
        return self
