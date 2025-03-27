from functools import cached_property, cache

from dataclasses import asdict, dataclass, field
from typing import ClassVar

import torch
import torchaudio


@cache
def _load_segment_audio(
    audio_path: str,
    new_freq: int = 24_000,
    load_kwargs: dict = {},
    resample_kwargs: dict = {},
) -> torch.Tensor:
    """
    Load audio from a file path and resample it to the specified frequency.

    Args:
        audio_path: Path to the audio file to load
        new_freq: Target sample rate to resample to (default: 24_000 Hz)
        # allowing kwargs for each of these as there are a lot of options and seems like the audio is kinda wonky
        load_kwargs: Additional keyword arguments to pass to torchaudio.load
        resample_kwargs: Additional keyword arguments to pass to torchaudio.functional.resample

    Returns:
        torch.Tensor: Resampled audio tensor with shape (num_samples,)
    """

    audio_tensor, sample_rate = torchaudio.load(audio_path, **load_kwargs)
    return torchaudio.functional.resample(
        waveform=audio_tensor.squeeze(0),
        orig_freq=sample_rate,
        new_freq=new_freq,
        **resample_kwargs,
    )


@dataclass
class Segment:
    """
    defining Segment here rather than import `csm.generator.Segment` to allow extending for easier caching of segments
    """

    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor

    def __hash__(self) -> int:
        return hash((self.speaker, self.text))


class _SID:
    _next_id: ClassVar[int] = 0

    @classmethod
    def get_next_id(cls) -> int:
        """
        Generate the next sequential speaker ID. Use this so that each speaker has a unique ID that is consistent/serial
        """
        cls._next_id += 1
        return cls._next_id - 1


@dataclass
class Speaker:
    """Participant in a conversation."""

    name: str
    description: str = ""
    # Make speaker_id a field with a default_factory to get the next ID
    speaker_id: int = field(default_factory=_SID.get_next_id)

    @classmethod
    def from_data(cls, data: dict | list[dict] | None = None) -> list["Speaker"]:
        """Convert the data to a list of speakers"""

        def _make(d):
            if isinstance(d, Speaker):
                return d
            elif isinstance(d, dict):
                return cls(**d)
            raise ValueError(f"Invalid data type: {type(d)}")

        # if data is None, return an empty list
        if data is None:
            return []

        data = [data] if isinstance(data, dict) else data

        return [_make(d) for d in data]

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
        # return f"{self.speaker.name}: {self.text}"
        return f"{self.speaker}: {self.text}"

    # if i cache this, i assume that the audio is not going to change...
    # probably a fine assumption given it loads the segment from the audio file?
    @cached_property
    def segment(self) -> Segment:
        """
        Convert this conversation turn to a CSM Segment for use as context.

        Returns:
            A Segment object containing speaker ID, text, and audio
        """
        if self.audio is None and self.audio_path is None:
            raise ValueError("Cannot convert to CSM Segment without audio")

        audio = self.audio
        if audio is None:
            audio = _load_segment_audio(self.audio_path)

        return Segment(speaker=self.speaker.speaker_id, text=self.text, audio=audio)


@dataclass
class TextOptions:
    max_tokens: int  # ~300?
    temperature: float = 0.8

    def asdict(self, **kwargs) -> dict:
        """
        Convert the TextOptions to a dictionary with optional additional parameters.
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
