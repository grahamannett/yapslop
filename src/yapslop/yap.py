import asyncio
from functools import partial
import random
import re
from collections import deque
from dataclasses import dataclass
from itertools import count
from os import makedirs, getenv
from typing import Any, AsyncGenerator, Coroutine, Iterable

from yapslop.convo_dto import ConvoTurn, Speaker, TextOptions, Segment
from yapslop.convo_helpers import generate_speaker, make_messages
from yapslop.providers.yaproviders import TextProvider, AudioProvider
from yapslop.utils.console import info, debug, rule

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


@dataclass
class HTTPConfig:
    base_url: str = getenv("LLM_BASE_URL", "http://localhost:11434")


class ConvoTextMixin:
    text_provider: TextProvider
    text_options: TextOptions

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


class ConvoManager(ConvoTextMixin):
    """
    Manages a simulated conversation between multiple speakers using a language model.
    """

    convo_system_prompt: str
    convo_speaker_desc: str
    speakers: list[Speaker]
    audio_provider: AudioProvider

    def __init__(
        self,
        text_provider: TextProvider,
        audio_provider: AudioProvider | None = None,
        n_speakers: int = 2,
        speakers: list[Speaker] | list[dict] | None = None,
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

        self.speakers = Speaker.from_data(speakers)
        self.n_speakers = n_speakers

        self.history: list[ConvoTurn] = []
        self.limit_context_turns = limit_context_turns

        # context queue is used for audio generation to have consistent sounds
        self._context_queue = deque(maxlen=self.limit_context_turns)

        if self.audio_output_dir:
            makedirs(self.audio_output_dir, exist_ok=True)

    @property
    def context_queue(self):
        return list(self._context_queue)

    def _get_turn_iter(self, num_turns: int | Iterable):
        # allow for the number of turns to be a count, a range, or an iterable
        match num_turns:
            case -1:
                num_turns = count()
            case int():
                num_turns = range(num_turns)
            case Iterable():
                num_turns = num_turns
        return num_turns

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
            segment: Segment = turn.segment
            self._context_queue.append(segment)

        return turn

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

        # --- Generate Text For the Turn if none provided
        if text is None:
            convo_system_prompt = simulator_system_prompt.format(
                convo_speaker_desc=self.convo_speaker_desc,
                convo_history="\n".join([str(turn) for turn in self.history]),
            )

            msgs = make_messages(f"{speaker.name}:", system=convo_system_prompt)

            text: str = await self.text_provider.chat_oai(messages=msgs, model_options=self.text_options)
            text = self._clean_generated_text(text=text, speaker=speaker)

        turn = ConvoTurn(speaker=speaker, text=text)
        context: list[Any] = self.context_queue

        async def _gen_audio(text_):
            return await self.audio_provider.generate_audio(
                text=text_,
                speaker_id=turn.speaker.speaker_id,
                context=context,
                max_audio_length_ms=max_audio_length_ms,
            )

        # --- Generate Audio For the Turn
        if do_audio_generate and self.audio_provider and turn.text:
            try:
                audio = await _gen_audio(turn.text)
            except Exception:
                print(f"Error generating text: {turn.text}, shortening text")
                audio = await _gen_audio(await self._create_shorter_text(turn.text))

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

        num_turns = self._get_turn_iter(num_turns)

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


class ConvoManangerQueue(ConvoManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.audio_queue = asyncio.Queue()
        self.text_queue = asyncio.Queue()

    async def text_producer(self, num_turns: int | Iterable):
        num_turns = self._get_turn_iter(num_turns)

        for turn in num_turns:
            speaker = self.select_next_speaker()

            convo_system_prompt = simulator_system_prompt.format(
                convo_speaker_desc=self.convo_speaker_desc,
                convo_history="\n".join([str(turn) for turn in self.history]),
            )

            msgs = make_messages(f"{speaker.name}:", system=convo_system_prompt)

            text: str = await self.text_provider.chat_oai(messages=msgs, model_options=self.text_options)
            text = self._clean_generated_text(text=text, speaker=speaker)

            turn = ConvoTurn(speaker=speaker, text=text)

            await self.text_queue.put(turn)
            info(f"Generated text for turn {turn.turn_idx}")

        await self.text_queue.put(None)

    async def audio_producer(self, max_audio_length_ms: int = 90_000, to_thread: bool = True):
        while True:
            turn = await self.text_queue.get()

            if turn is None:
                break

            func = self.audio_provider.sync_generate_audio
            if to_thread:

                def pfunc(text, speaker_id, context, max_audio_length_ms):
                    return asyncio.to_thread(func, text, speaker_id, context, max_audio_length_ms)

                func = pfunc

            debug(f">>Got text for turn {turn.turn_idx}")

            audio = func(
                text=turn.text,
                speaker_id=turn.speaker.speaker_id,
                context=self.context_queue,
                max_audio_length_ms=max_audio_length_ms,
            )

            if isinstance(audio, Coroutine):
                audio = await audio

            debug(f"Generated audio for turn {turn.turn_idx} {len(audio)=}")
            await self.audio_queue.put(audio)

    async def run(self, num_turns: int = 3, initial_text: str | None = None, to_thread: bool = False, **kwargs):
        self.initial_text = initial_text
        info(f"Running with {num_turns} turns, {to_thread=}")

        await asyncio.gather(
            self.text_producer(num_turns=num_turns),
            self.audio_producer(to_thread=to_thread),
        )
