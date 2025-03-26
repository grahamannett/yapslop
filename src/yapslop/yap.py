import os
import random
import re
from collections import deque
from dataclasses import dataclass
from itertools import count
from typing import AsyncGenerator, Iterable

import httpx
import torch
import torchaudio

from yapslop.audio_helpers import load_audio
from yapslop.convo_dto import ConvoTurn, Speaker, TextOptions
from yapslop.convo_helpers import MessageType, generate_speaker, make_messages
from yapslop.generator import Segment, load_csm_1b
from yapslop.yap_common import DEVICE

# --- prompts
# - simulator_system_prompt: the system prompt for the simulator
# - shorter_system_prompt: the system prompt to shorten text

simulator_system_prompt = """You are simulating a conversation between the following characters:{convo_speaker_desc}

Follow these rules:
1. Respond ONLY as the designated speaker for each turn
2. Stay in character at all times and don't refer to yourself in the third person
3. Keep responses concise (similar in length to the prior response length) and natural-sounding
4. Don't narrate actions or use quotation marks

Previous Conversation:
{convo_history}
"""

shorter_system_prompt = """Create a shorter version of the following text.
Keep the same meaning and return ONLY the text in a more concise form"""

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


def turn_to_segment(turn: ConvoTurn) -> Segment:
    """
    Convert this conversation turn to a CSM Segment for use as context.

    Loads audio from file if not already in memory.

    Returns:
        A Segment object containing speaker ID, text, and audio

    Raises:
        ValueError: If neither audio nor audio_path is available
    """
    audio = turn.audio
    if audio is None and turn.audio_path:
        audio = load_audio(turn.audio_path)

    if (audio is None) and (turn.audio_path is None):
        raise ValueError("Cannot convert to CSM Segment without audio")

    return Segment(speaker=turn.speaker.speaker_id, text=turn.text, audio=audio)


@dataclass
class HTTPConfig:
    base_url: str = "http://localhost:11434"


@add_provider("text")
class TextProvider:
    """Base class for model providers, can switch to using pydantic-ai/vllm/etc later"""

    client: httpx.AsyncClient
    model_name: str = "gemma3:latest"

    def _from_resp(self, resp: httpx.Response) -> dict:
        """Parse and validate response from API"""
        resp.raise_for_status()
        return resp.json()

    async def __call__(self, *args, **kwargs):
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


class ConvoManager:
    """
    Manages a simulated conversation between multiple speakers using a language model.
    """

    convo_system_prompt: str
    convo_speaker_desc: str

    def __init__(
        self,
        text_provider: TextProvider,
        audio_provider: AudioProvider | None = None,
        n_speakers: int = 2,
        speakers: list[Speaker] = [],
        max_tokens: int = 300,
        temperature: float = 0.7,
        audio_output_dir: str | None = None,
        limit_context_turns: int | None = 3,  # limit either the text length or the audio context length
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

        self.history: list[ConvoTurn] = []
        self.limit_context_turns = limit_context_turns

        # context queue is used for audio generation to have consistent sounds
        self.context_queue = deque(maxlen=self.limit_context_turns)

        if self.audio_output_dir:
            os.makedirs(self.audio_output_dir, exist_ok=True)

    def _clean_generated_text(self, text: str, speaker: Speaker) -> str:
        """
        Remove the speaker name from the text if it's at the beginning of the text.

        Args:
            text: Text to clean
            speaker: Speaker whose name to remove

        Returns:
            Cleaned text with speaker name removed if present
        """
        text = re.sub(f"^{speaker.name}[:|\n]\\s*", "", text).strip()
        return text

    def _post_turn(self, turn: ConvoTurn, save_audio: bool) -> ConvoTurn:
        """
        Process a turn after generation - save audio if needed and update context.

        Args:
            turn: The turn to process
            save_audio: Whether to save the audio to disk

        Returns:
            The processed turn
        """
        if save_audio and self.audio_provider and self.audio_output_dir and turn.audio != None:
            audio_filename = f"turn_{len(self.history)}_speaker_{turn.speaker.speaker_id}.wav"
            turn.audio_path = f"{self.audio_output_dir}/{audio_filename}"
            self.audio_provider.save_audio(turn.audio, turn.audio_path)

        if turn.audio is not None:
            self.context_queue.append(turn_to_segment(turn))

        return turn

    async def _create_shorter_text(self, text: str) -> str:
        """
        Create a shorter version of the given text while preserving meaning.

        Args:
            text: Text to shorten

        Returns:
            Shortened version of the text
        """
        messages = make_messages(shorter_system_prompt, text)
        return await self.text_provider.chat_oai(messages=messages, model_options=self.text_options)

    async def setup_speakers(
        self, n_speakers: int | None = None, speakers: list[Speaker] | None = None
    ) -> list[Speaker]:
        """
        Generate a list of speakers for the conversation.

        Args:
            n_speakers: Number of speakers to generate
            speakers: Optional list of pre-defined speakers

        Returns:
            List of speakers for the conversation
        """
        n_speakers = n_speakers or self.n_speakers
        speakers = speakers or []

        # if we have pre-defined speakers, add them to the list
        if self.speakers:
            speakers += self.speakers

        for _ in range(len(speakers), n_speakers):
            speakers.append(await generate_speaker(self.text_provider.chat_ollama, speakers))

        # setup the system prompt that contains the speaker descriptions and rules
        self.convo_speaker_desc = ""
        for speaker in speakers:
            self.convo_speaker_desc += f"\n- {speaker.name} : {speaker.description}"

        # update the speakers to be the new speakers (+ potentially pre-defined speakers)
        self.speakers = speakers
        return speakers

    def select_next_speaker(self, speaker: Speaker | None = None) -> Speaker:
        """
        Select the next speaker for the conversation.

        Args:
            speaker: Optional speaker to force as next speaker

        Returns:
            Selected speaker for next turn
        """
        if speaker:
            return speaker

        if not self.history:
            return random.choice(self.speakers)

        return random.choice([s for s in self.speakers if s != self.history[-1].speaker])

    async def generate_turn(
        self,
        speaker: Speaker | None = None,
        text: str | None = None,
        do_audio_generate: bool = True,
        save_audio: bool = True,
        max_audio_length_ms: int = 90_000,
    ) -> ConvoTurn:
        """
        Generate the next turn in the conversation.

        Args:
            speaker: Optional speaker to force as the next speaker
            text: Optional text to force as the next text
            do_audio_generate: Whether to generate audio for this turn
            save_audio: Whether to save the audio for this turn
            max_audio_length_ms: Maximum audio length in milliseconds

        Returns:
            ConvoTurn object containing the generated text and optionally audio
        """
        if not speaker:
            speaker = self.select_next_speaker()

        turn = ConvoTurn(speaker=speaker)

        # --- Generate Text For the Turn if none provided
        if text is None:
            convo_history = "\n".join([str(turn) for turn in self.history])

            convo_system_prompt = simulator_system_prompt.format(
                convo_speaker_desc=self.convo_speaker_desc,
                convo_history=convo_history,
            )

            msgs = make_messages(f"{speaker.name}:", system=convo_system_prompt)

            text = await self.text_provider.chat_oai(messages=msgs, model_options=self.text_options)
            text = self._clean_generated_text(text=text, speaker=speaker)

        turn.text = text

        # --- Generate Audio For the Turn
        if do_audio_generate and self.audio_provider:
            context = list(self.context_queue)

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
        num_turns: int | Iterable,
        initial_text: str | None = None,
        initial_speaker: Speaker | None = None,
        do_audio_generate: bool = True,
        save_audio: bool = True,
        max_audio_length_ms: int = 90_000,
    ) -> AsyncGenerator[ConvoTurn, None]:
        """
        Generate a conversation with the specified number of turns as an async generator.
        Yields each turn immediately after it's generated for real-time processing.

        Args:
            num_turns: Number of turns to generate, -1 for infinite
            initial_text: Optional text to start the conversation with
            initial_speaker: Optional speaker for the initial phrase
            do_audio_generate: Whether to generate audio for each turn
            save_audio: Whether to save the audio for each turn
            max_audio_length_ms: Maximum audio length in milliseconds

        Yields:
            ConvoTurn objects one at a time as they are generated
        """

        # allow for the initial text and speaker to be passed in
        text = initial_text
        speaker = initial_speaker

        # allow for the number of turns to be a count, a range, or an iterable
        match num_turns:
            case -1:
                num_turns = count()
            case int():
                num_turns = range(num_turns)
            case Iterable():
                num_turns = num_turns

        for t_idx in num_turns:
            turn = await self.generate_turn(
                text=text,
                speaker=speaker,
                save_audio=save_audio,
                do_audio_generate=do_audio_generate,
                max_audio_length_ms=max_audio_length_ms,
            )
            turn.turn_idx = t_idx
            yield turn
            # reset the phrase and speaker for the next turn
            text = None
            speaker = None
