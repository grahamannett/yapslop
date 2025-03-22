from functools import wraps
import os
import random
from dataclasses import dataclass
from typing import Literal

import httpx
import torch
import torchaudio

from yapslop.convo_dto import ConvoTurn, Speaker, TextOptions
from yapslop.convo_helpers import MessageType, generate_speaker_dict
from yapslop.generator import load_csm_1b, Segment

DEVICE: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"
simulator_system_prompt = """You are simulating a conversation between the following characters:
{speakers_desc}

Follow these rules:
1. Respond ONLY as the designated speaker for each turn
2. Stay in character at all times and don't refer to yourself in the third person
3. Keep responses concise (similar in length to the prior response length) and natural-sounding
4. Don't narrate actions or use quotation marks
"""

shorter_system_prompt = """Create a shorter version of the following text.
Keep the same meaning and return ONLY the text in a more concise form"""

# using _Providers as a registry, can store the type of provider, or use like "ollama" in future
# avoid using too deep a structure for all of this as will make future threading/multiprocessing easier
_Providers = {}


def add_provider(name: str, use_dc: bool = True):
    def decorator(func):  # move dataclass to this for slightly clearer code
        _Providers[name] = dataclass(func) if use_dc else func
        return func

    return decorator


def ProvidersSetup(configs: dict[str, dict]):
    return [_Providers[key](**config) for key, config in configs.items()]


def make_convo_system_prompt(speakers: list[Speaker]) -> str:
    def format_speaker(speaker: Speaker) -> str:
        prompt = f"{speaker.name}"
        if speakers_desc := getattr(speaker, "description", None):
            prompt += f" ({speakers_desc})"
        if speakers_style := getattr(speaker, "speaking_style", None):
            prompt += f"\nSpeaking style: {speakers_style}"
        return prompt

    speakers_desc = "\n\n".join(format_speaker(speaker) for speaker in speakers)
    return simulator_system_prompt.format(speakers_desc=speakers_desc)


@dataclass
class HTTPConfig:
    base_url: str = "http://localhost:11434"


@add_provider("text")
class TextProvider:
    """Base class for model providers, can switch to using pydantic-ai/vllm/etc later"""

    client: httpx.AsyncClient
    model_name: str = "gemma3:latest"

    def _from_resp(self, resp: httpx.Response) -> dict:
        resp.raise_for_status()
        return resp.json()

    async def __call__(self, *args, **kwargs):
        return await self.chat_oai(*args, **kwargs)

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
            context_turns: Optional list of previous ConvoTurn objects with audio
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
        """Save the generated audio to a file."""
        torchaudio.save(file_path, audio.unsqueeze(0).cpu(), self.sample_rate)


class ConvoManager:
    """
    Manages a simulated conversation between multiple speakers using a language model.
    """

    def __init__(
        self,
        text_provider: TextProvider,
        audio_provider: AudioProvider | None = None,
        n_speakers: int = 2,
        speakers: list[Speaker] = [],
        system_prompt: str = "",
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

        if self.audio_output_dir:
            os.makedirs(self.audio_output_dir, exist_ok=True)

    def _cleanup_text_turn(self, text: str, speaker: Speaker) -> str:
        """Remove the speaker name from the text if it's at the beginning of the text"""
        if text.startswith(f"{speaker.name}:"):
            text = text[len(f"{speaker.name}:") :].strip()
        return text

    def _post_turn(self, turn: ConvoTurn, save_audio: bool) -> ConvoTurn:
        """Save the audio for the turn if the audio provider is set and the audio output dir is set"""

        if save_audio and self.audio_provider and self.audio_output_dir and turn.audio != None:
            audio_filename = f"turn_{len(self.history)}_speaker_{turn.speaker.speaker_id}.wav"
            turn.audio_path = f"{self.audio_output_dir}/{audio_filename}"
            self.audio_provider.save_audio(turn.audio, turn.audio_path)
        return turn

    async def _create_shorter_text(self, text: str) -> str:
        msg = [
            {"role": "system", "content": shorter_system_prompt},
            {"role": "user", "content": text},
        ]
        return await self.text_provider.chat_oai(messages=msg, model_options=self.text_options)

    def _create_msgs_for_next_turn(
        self, next_speaker: Speaker | None = None
    ) -> list[dict[str, str]]:
        """
        Create the prompt for the next turn in the conversation.

        Args:
            next_speaker: The speaker who will generate the next turn

        Returns:
            List of message dictionaries for the API call
        """

        if self.history:
            prev_convo = "\n".join([str(turn) for turn in self.history])
            system_prompt = f"{self.system_prompt}\nPrevious Conversation:\n{prev_convo}"

        msgs = [{"role": "system", "content": system_prompt}]

        if next_speaker:
            msgs += [{"role": "user", "content": f"{next_speaker.name}:"}]

        return msgs

    async def setup_speakers(
        self, n_speakers: int | None = None, speakers: list[Speaker] = []
    ) -> list[Speaker]:
        """
        Generate a list of speakers for the conversation.  Allows you to pass in speakers and generate more
        """

        n_speakers = n_speakers or self.n_speakers

        if self.speakers:
            speakers += self.speakers

        for _ in range(len(speakers), n_speakers):
            speaker = await generate_speaker_dict(self.text_provider, speakers=speakers)
            speakers.append(Speaker(**speaker))

        self.speakers = speakers
        self.system_prompt = self.system_prompt or make_convo_system_prompt(speakers)
        return speakers

    def select_next_speaker(self, speaker: Speaker | None = None) -> Speaker:
        """
        Select the next speaker for the conversation.
        By default, rotates through speakers in order, avoiding consecutive turns.
        """

        if speaker:
            return speaker

        if not self.history:
            return random.choice(self.speakers)

        return random.choice([s for s in self.speakers if s != self.history[-1].speaker])

    def _get_context(self) -> list[Segment]:
        context_turns = [t for t in self.history[:-1] if t.audio is not None]

        if self.limit_context_turns:
            context_turns = context_turns[-self.limit_context_turns :]

        # Convert ConvoTurn objects to CSM Segments if context is provided
        context = []
        if context_turns:
            for turn in context_turns:
                if turn.audio is not None:
                    context.append(turn.to_segment())
        return context

    async def generate_turn(
        self,
        speaker: Speaker,
        text: str | None = None,
        do_audio_generate: bool = True,
        save_audio: bool = True,
        max_audio_length_ms: int = 90_000,
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
        turn = ConvoTurn(speaker=speaker)

        if text is None:
            msgs = self._create_msgs_for_next_turn(speaker)
            text = await self.text_provider.chat_oai(messages=msgs, model_options=self.text_options)
            text = self._cleanup_text_turn(text=text, speaker=speaker)

        turn.text = text

        # --- Generate Audio For the Turn
        if do_audio_generate and self.audio_provider:
            context = self._get_context()
            try:
                audio = self.audio_provider.generate_audio(
                    text=turn.text,
                    speaker_id=turn.speaker.speaker_id,
                    context=context,
                    max_audio_length_ms=max_audio_length_ms,
                )
            except Exception:
                print(f"Error generating text: {turn.text}, shortening text")
                turn.text = await self._create_shorter_text(turn.text)
                audio = self.audio_provider.generate_audio(
                    text=turn.text,
                    speaker_id=turn.speaker.speaker_id,
                    context=context,
                    max_audio_length_ms=max_audio_length_ms,
                )

            turn.audio = audio
            turn = self._post_turn(turn, save_audio=save_audio)

        self.history.append(turn)

        return turn

    async def generate_convo_stream(
        self,
        num_turns: int,
        initial_text: str | None = None,
        initial_speaker: Speaker | None = None,
        do_audio_generate: bool = True,
        save_audio: bool = True,
        max_audio_length_ms: int = 90_000,
    ):
        """
        Generate a conversation with the specified number of turns as an async generator.
        Yields each turn immediately after it's generated for real-time processing.

        Args:
            num_turns: Number of turns to generate
            initial_text: Optional text to start the conversation with
            initial_speaker: Optional speaker for the initial phrase
            do_audio_generate: Whether to generate audio for each turn
            save_audio: Whether to save the audio for each turn

        Yields:
            ConvoTurn objects one at a time as they are generated
        """

        # allow for the initial text and speaker to be passed in
        text = initial_text
        speaker = initial_speaker

        while num_turns != 0:
            speaker = self.select_next_speaker(speaker=speaker)
            turn = await self.generate_turn(
                text=text,
                speaker=speaker,
                save_audio=save_audio,
                do_audio_generate=do_audio_generate,
                max_audio_length_ms=max_audio_length_ms,
            )
            yield turn
            # reset the phrase and speaker if passed in for next turn
            num_turns -= 1
            text = None
            speaker = None

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
