import asyncio
import os
from pathlib import Path
import random
from dataclasses import asdict, dataclass, field
from typing import ClassVar

import httpx
import torch
import torchaudio
from huggingface_hub import hf_hub_download

from csm.generator import Segment, load_csm_1b
from yapslop.convo_helpers import MessageType, generate_speaker_dict

device = "cuda" if torch.cuda.is_available() else "cpu"
system_prompt_template = """
You are simulating a conversation between the following characters:
{speakers_desc}

Follow these rules:
1. Respond ONLY as the designated speaker for each turn
2. Stay in character at all times
3. Keep responses concise and natural-sounding
4. Don't narrate actions or use quotation marks
5. Don't refer to yourself in the third person
6. Keep the response length similar to the other responses.
"""


def load_audio(audio_path: str, new_freq: int = 24_000) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    return torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=new_freq
    )


def make_convo_system_prompt(speakers: list["Speaker"]) -> str:
    def format_speaker(speaker: "Speaker") -> str:
        prompt = f"{speaker.name}"
        if speakers_desc := getattr(speaker, "description", None):
            prompt += f" ({speakers_desc})"
        if speakers_style := getattr(speaker, "speaking_style", None):
            prompt += f"\nSpeaking style: {speakers_style}"
        return prompt

    speakers_desc = "\n\n".join(format_speaker(speaker) for speaker in speakers)
    return system_prompt_template.format(speakers_desc=speakers_desc)


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


@dataclass
class TextProvider:
    """Base class for model providers, switch to using pydantic-ai/vllm/etc later"""

    client: httpx.AsyncClient
    model_name: str

    def _from_resp(self, resp: httpx.Response) -> dict:
        resp.raise_for_status()
        return resp.json()

    async def chat_oai(
        self, messages: MessageType, model_options: TextOptions | dict = {}, **kwargs
    ) -> str:
        """Generate text using OpenAI API endpoint"""
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

        The reason we may prefer this over the openai endpoint is there is variability in the tool use
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


@dataclass
class AudioProvider:
    repo_id: str = "sesame/csm-1b"
    device: str = "cuda"

    def __post_init__(self):
        self.model_path = hf_hub_download(repo_id=self.repo_id, filename="ckpt.pt")
        self.generator = load_csm_1b(self.model_path, device=self.device)

    @property
    def sample_rate(self) -> int:
        return self.generator.sample_rate

    def generate_audio(
        self,
        text: str,
        speaker_id: int,
        context_turns: list[ConvoTurn] = [],
        max_audio_length_ms: int | None = None,
    ):
        """
        Generate audio for the given text and speaker.

        Args:
            text: The text to convert to speech
            speaker_id: Speaker ID for the CSM model (0, 1, etc.)
            context_turns: Optional list of previous ConvoTurn objects with audio
            max_audio_length_ms: Maximum audio length in milliseconds

        Returns:
            torch.Tensor containing the generated audio
        """
        # Convert ConvoTurn objects to CSM Segments if context is provided
        context_segments = []
        if context_turns:
            for turn in context_turns:
                if turn.audio is not None:
                    context_segments.append(turn.to_segment())

        generate_kwargs = {
            "text": text,
            "speaker": speaker_id,
            "context": context_segments,
            **({"max_audio_length_ms": max_audio_length_ms} if max_audio_length_ms else {}),
        }

        audio = self.generator.generate(**generate_kwargs)

        return audio

    def save_audio(self, audio: torch.Tensor, file_path: str):
        """Save the generated audio to a file."""
        torchaudio.save(file_path, audio.unsqueeze(0).cpu(), self.sample_rate)


class ConvoManager:
    """
    Manages a simulated conversation between multiple speakers using a language model.
    """

    system_prompt: str

    def __init__(
        self,
        text_provider: TextProvider,
        audio_provider: AudioProvider | None = None,
        n_speakers: int = 2,
        speakers: list[Speaker] = [],
        system_prompt: str | None = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
        audio_output_dir: str | None = None,
        limit_context_turns: int
        | None = 3,  # limit either the text length or the audio context length
    ):
        """
        Initialize the conversation manager.

        Args:
            text_provider: Model provider to use for text generation
            audio_provider: Optional AudioProvider for speech synthesis
            n_speakers: Number of speakers to include in the conversation
            speakers: Optional list of pre-defined speakers. If empty, speakers will be generated
            system_prompt: Optional system prompt to guide the conversation
            max_tokens: Maximum tokens to generate per turn
            temperature: Temperature setting for generation (higher = more random)
            audio_output_dir: Directory to save generated audio files
            limit_context_turns: Maximum number of previous turns to use as context for audio generation
        """

        self.text_provider = text_provider
        self.text_options = TextOptions(max_tokens=max_tokens, temperature=temperature)

        self.audio_provider = audio_provider
        self.audio_output_dir = audio_output_dir

        # Default system prompt if none provided
        self.speakers = speakers
        self.n_speakers = n_speakers
        self.system_prompt = system_prompt

        self.history: list[ConvoTurn] = []
        self.limit_context_turns = limit_context_turns

    def _setup_audio_provider(self):
        if self.audio_output_dir:
            os.makedirs(self.audio_output_dir, exist_ok=True)

    def _cleanup_text_turn(self, text: str, speaker: Speaker) -> str:
        if text.startswith(f"{speaker.name}:"):
            text = text[len(f"{speaker.name}:") :].strip()
        return text

    def _post_turn(self, turn: ConvoTurn, save_audio: bool) -> ConvoTurn:
        if save_audio and self.audio_provider and self.audio_output_dir:
            audio_filename = f"turn_{len(self.history)}_speaker_{turn.speaker.speaker_id}.wav"
            turn.audio_path = f"{self.audio_output_dir}/{audio_filename}"
            self.audio_provider.save_audio(turn.audio, turn.audio_path)
        return turn

    def _create_prompt_for_next_turn(
        self, next_speaker: Speaker | None = None
    ) -> list[dict[str, str]]:
        """
        Create the prompt for the next turn in the conversation.

        Args:
            next_speaker: The speaker who will generate the next turn

        Returns:
            List of message dictionaries for the API call
        """
        system_prompt = self.system_prompt

        if self.history:
            system_prompt += "\n---\nPrevious conversation Turns:\n"

            for turn in self.history:
                system_prompt += f"{turn.speaker.name}: {turn.text}"

        msgs = [{"role": "system", "content": system_prompt}]
        if next_speaker:
            msgs.append(
                {
                    "role": "user",
                    "content": f"{next_speaker.name}:",
                }
            )

        return msgs

    async def setup_speakers(
        self, n_speakers: int | None = None, speakers: list[Speaker] = []
    ) -> list[Speaker]:
        """
        Generate a list of speakers for the conversation.  Allows you to pass in speakers or
        """
        # allow for passing in speakers or use the existing speakers
        n_speakers = n_speakers or self.n_speakers
        if self.speakers:
            speakers += self.speakers

        for _ in range(len(speakers), n_speakers):
            speaker = await generate_speaker_dict(self.text_provider, speakers=speakers)
            speakers.append(Speaker(**speaker))

        self.speakers = speakers
        self.system_prompt = self.system_prompt or make_convo_system_prompt(speakers)
        return speakers

    def select_next_speaker(self) -> Speaker:
        """
        Select the next speaker for the conversation.
        By default, rotates through speakers in order, avoiding consecutive turns.
        """
        if not self.history:
            return random.choice(self.speakers)

        return random.choice([s for s in self.speakers if s != self.history[-1].speaker])

    async def generate_turn(
        self,
        speaker: Speaker,
        text: str | None = None,
        do_audio_generate: bool = True,
        save_audio: bool = True,
    ) -> ConvoTurn:
        """
        Generate the next turn in the conversation.

        Args:
            speaker: Optional speaker to force as the next speaker
            generate_audio: Whether to generate audio for this turn

        Returns:
            ConvoTurn object containing the generated text and optionally audio
        """
        # --- Generate Text For the Turn, Optionally Provide the Text

        if text is None:
            msgs = self._create_prompt_for_next_turn(speaker)
            text = await self.text_provider.chat_oai(messages=msgs, model_options=self.text_options)

            text = self._cleanup_text_turn(text=text, speaker=speaker)

        turn = ConvoTurn(speaker=speaker, text=text)

        # --- Generate Audio For the Turn
        if do_audio_generate and self.audio_provider:
            # Get recent context turns with audio (limit to 3 for performance)
            context_turns = [t for t in self.history[:-1] if t.audio is not None]

            if self.limit_context_turns:
                context_turns = context_turns[-self.limit_context_turns :]

            turn.audio = self.audio_provider.generate_audio(
                text=turn.text,
                speaker_id=turn.speaker.speaker_id,
                context_turns=context_turns,
            )

            turn = self._post_turn(turn, save_audio=save_audio)

        self.history.append(turn)

        return turn

    async def generate_convo_text_stream(
        self,
        num_turns: int,
        initial_phrase: str | None = None,
        initial_speaker: Speaker | None = None,
        do_audio_generate: bool = True,
        save_audio: bool = True,
    ):
        """
        Generate a conversation with the specified number of turns as an async generator.
        Yields each turn immediately after it's generated for real-time processing.

        Args:
            num_turns: Number of turns to generate
            initial_phrase: Optional text to start the conversation with
            initial_speaker: Optional speaker for the initial phrase
            do_audio_generate: Whether to generate audio for each turn
            save_audio: Whether to save the audio for each turn

        Yields:
            ConvoTurn objects one at a time as they are generated
        """
        # Add initial phrase if provided
        if initial_phrase:
            speaker = initial_speaker or self.select_next_speaker()
            turn = await self.generate_turn(
                text=initial_phrase,
                speaker=speaker,
                save_audio=save_audio,
                do_audio_generate=do_audio_generate,
            )
            yield turn

        # Generate the rest of the conversation one turn at a time
        for _ in range(num_turns):
            turn = await self.generate_turn(
                speaker=self.select_next_speaker(),
                do_audio_generate=do_audio_generate,
                save_audio=save_audio,
            )
            yield turn

    def save_combined_audio(
        self,
        output_path: str,
        turns: list[ConvoTurn] | None = None,
        add_silence_ms: int = 500,
    ) -> str:
        """
        Combine audio from multiple conversation turns into a single audio file with silence between each turn.

        Args:
            output_path: Path where the combined audio file will be saved
            turns: Optional list of ConvoTurn objects containing audio to combine. If None, uses self.history
            add_silence_ms: Milliseconds of silence to insert between each turn. Defaults to 500ms

        Returns:
            str: Path to the saved combined audio file

        Raises:
            ValueError: If no audio provider is initialized or if no turns with audio are found
        """
        if self.audio_provider is None:
            raise ValueError("Audio provider not initialized")

        # Use provided turns or conversation history
        turns = turns or self.history

        # Extract audio tensors from turns that have audio
        audio_tensors = [turn.audio for turn in turns if turn.audio is not None]

        if not audio_tensors:
            raise ValueError("No turns with audio found")

        # Calculate silence samples
        silence_samples = int(self.audio_provider.sample_rate * add_silence_ms / 1000)
        silence = torch.zeros(silence_samples, device=self.audio_provider.device)

        # Combine audio tensors with silence between them
        combined_tensors = []
        for i, audio in enumerate(audio_tensors):
            combined_tensors.append(audio)
            # Add silence after each segment except the last
            if i < len(audio_tensors) - 1:
                combined_tensors.append(silence)

        # Concatenate all audio tensors along the time dimension
        final_audio = torch.cat(combined_tensors)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the combined audio
        torchaudio.save(
            output_path,
            final_audio.unsqueeze(0).cpu(),  # Add channel dimension and ensure on CPU
            self.audio_provider.sample_rate,
        )

        return output_path


# Example usage demonstration
async def demo(
    n_speakers: int = 2,
    num_turns: int = 5,
    initial_phrase: str = "Did you hear about that new conversational AI model that just came out?",
    audio_output_dir: str = "audio_output",
    cleanup_audio_dir: bool = True,
):
    if cleanup_audio_dir and (_dir := Path(audio_output_dir)):
        _dir.mkdir(parents=True, exist_ok=True)
        _ = [f.unlink() for f in _dir.glob("*.wav")]

    # demo showing one initial speaker but then generate the rest of the speakers
    speakers = [
        Speaker(
            name="Seraphina",
            description="Tech entrepreneur. Uses technical jargon, speaks confidently",
        ),
    ]

    # Create the text provider (using Ollama by default). Example using the streaming generator
    async with httpx.AsyncClient(base_url="http://localhost:11434") as client:
        audio_provider = AudioProvider(device=device)
        text_provider = TextProvider(client=client, model_name="gemma3:latest")

        convo_manager = ConvoManager(
            n_speakers=n_speakers,
            speakers=speakers,
            text_provider=text_provider,
            audio_provider=audio_provider,
            audio_output_dir=audio_output_dir,
        )

        await convo_manager.setup_speakers()

        initial_speaker = convo_manager.speakers[0]

        print("Streaming conversation in real-time\n" + "-" * 50)
        # Stream each turn as it's generated
        async for turn in convo_manager.generate_convo_text_stream(
            num_turns=num_turns,
            initial_phrase=initial_phrase,
            initial_speaker=initial_speaker,
        ):
            print(f"{turn}")
            if turn.audio_path:
                print(f"Audio saved to: {turn.audio_path}")

        print("-" * 50)

    # Save combined audio
    combined_audio_path = convo_manager.save_combined_audio(output_path="full_conversation.wav")
    print(f"Combined audio saved to: {combined_audio_path}")


if __name__ == "__main__":
    asyncio.run(demo())
