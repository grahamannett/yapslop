from functools import cached_property, cache

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
import torchaudio


@cache  # cant @cache with kwargs
def _load_segment_audio(
    audio_path: str,
    new_freq: int = 24_000,
    normalize: bool = True,
    channels_first: bool = True,
    buffer_size: int = 4096,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
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

    audio_tensor, sample_rate = torchaudio.load(
        audio_path,
        normalize=normalize,
        channels_first=channels_first,
        buffer_size=buffer_size,
    )

    return torchaudio.functional.resample(
        waveform=audio_tensor.squeeze(0),
        orig_freq=sample_rate,
        new_freq=new_freq,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        resampling_method=resampling_method,
        beta=beta,
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

    # @classmethod
    # def from_audio_path(cls, audio_path: str, speaker: int, text: str) -> "Segment":
    #     audio = _load_segment_audio(audio_path)
    #     return cls(speaker=speaker, text=text, audio=audio)


class ClassSequentialID:
    _next_id: ClassVar[int] = 0

    @classmethod
    def get_next_id(cls) -> int:
        """
        Generate the next sequential speaker ID. Use this so that each speaker has a unique ID that is consistent/serial
        """
        cls._next_id += 1
        return cls._next_id - 1


@dataclass
class BaseSpeaker:
    """
    Base class for speakers.

    This is used to generate the schema for the speaker.
    Might be better to name this one `Speaker` and have a `ConvoSpeaker` that is used for the conversation.
    """

    name: str
    description: str


class SpeakerID(ClassSequentialID):
    pass


@dataclass
class Speaker(BaseSpeaker):
    """Participant in a conversation."""

    description: str = ""
    # Make speaker_id a field with a default_factory to get the next ID
    speaker_id: int = field(default_factory=SpeakerID.get_next_id)

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


class ConvoTurnID(ClassSequentialID):
    pass


@dataclass
class BaseConvoTurn:
    speaker: Speaker
    text: str | None = None
    audio: torch.Tensor | None = None
    audio_path: str | None = None
    turn_id: int | None = None

    @classmethod
    def from_audio_path(
        cls,
        audio_path: str,
        text: str = "",
        speaker: Speaker | None = None,
        path_has_info: bool = True,
    ) -> "BaseConvoTurn":
        """
        Create a BaseConvoTurn from an audio file path. If path_has_info is True, parse the turn and speaker info from the file name.

        Args:
            audio_path (str): Path to the audio file, expected to contain turn and speaker info (e.g. '.../turn_3_speaker_0.wav').
            text (str | None): Optional text.
            speaker (Speaker | None): Optional Speaker instance. If not provided, will be created from parsed info or must be provided if path_has_info is False.
            path_has_info (bool): If True, attempt to parse turn and speaker info from audio_path.

        Returns:
            BaseConvoTurn: An instance with audio loaded, and parsed turn_id and speaker if available.
        """
        turn_id = 0
        if path_has_info:
            parts = Path(audio_path).stem.split("_")

            for idx, part in enumerate(parts):
                if part == "turn":
                    turn_id = int(parts[idx + 1])
                elif part == "speaker":
                    speaker_id = int(parts[idx + 1])

            if speaker is None:
                speaker = Speaker(name=f"Speaker {speaker_id}", description="", speaker_id=speaker_id)

        if speaker is None:
            raise ValueError("Speaker must be provided if not parsed from path")

        return cls(
            speaker=speaker,
            text=text,
            audio=_load_segment_audio(audio_path),
            audio_path=audio_path,
            turn_id=turn_id,
        )


@dataclass
class ConvoTurn(BaseConvoTurn):
    """
    Represents a single turn in a conversation.
    """

    turn_id: int = field(default_factory=ConvoTurnID.get_next_id)

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
