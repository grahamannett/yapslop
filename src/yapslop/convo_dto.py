from dataclasses import asdict, dataclass, field
from typing import ClassVar

import torch
import torchaudio

from yapslop.generator import Segment


def load_audio(audio_path: str, new_freq: int = 24_000) -> torch.Tensor:
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
        cls._next_id += 1
        return cls._next_id - 1

    def __str__(self) -> str:
        return f"{self.name}"

    def asdict(self) -> dict:
        return asdict(self)


@dataclass
class ConvoTurn:
    """
    Represents a single turn in a conversation.
    """

    speaker: Speaker
    text: str
    # Add audio field to store generated audio
    audio: torch.Tensor | None = None
    audio_path: str | None = None

    def __str__(self) -> str:
        return f"{self.speaker.name}: {self.text}"

    def to_segment(self) -> Segment:
        """
        Convert this conversation turn to a CSM Segment for use as context
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
        return asdict(self) | kwargs

    # some of the ollama options are different that the openai options, i know ill have to deal with both of these later
    def oai_fmt(self):
        return self

    def ollama_fmt(self):
        return self
