import os
import random
import asyncio
from dataclasses import dataclass
import httpx
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from csm.generator import load_csm_1b, Segment

system_prompt = """
You are simulating a conversation between the following characters:
{speakers_desc}

Follow these rules:
1. Respond ONLY as the designated speaker for each turn
2. Stay in character at all times
3. Keep responses concise and natural-sounding
4. Don't narrate actions or use quotation marks
5. Don't refer to yourself in the third person
"""


def load_audio(audio_path: str, new_freq: int = 24_000) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    return torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=new_freq
    )


def make_system_prompt(speakers: list["Speaker"]) -> str:
    def make_char_prompt(speaker: "Speaker") -> str:
        prompt = f"{speaker.name}"
        if speaker.description:
            prompt += f" ({speaker.description})"
        if speaker.personality:
            prompt += f"\nPersonality: {speaker.personality}"
        if speaker.speaking_style:
            prompt += f"\nSpeaking style: {speaker.speaking_style}"
        return prompt

    return system_prompt.format(speakers_desc="\n".join([make_char_prompt(s) for s in speakers]))


def get_conversation_as_string(history: list["ConvoTurn"]) -> str:
    return "\n".join([str(turn) for turn in history])


@dataclass
class Speaker:
    """
    Represents a participant in a conversation.
    """

    name: str
    description: str = ""
    personality: str = ""
    speaking_style: str = ""
    # Add speaker_id for CSM model integration
    speaker_id: int = 0

    def __str__(self) -> str:
        return f"{self.name}"


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


class TextModelProvider:
    """Base class for model providers, switch to using pydantic-ai/vllm/etc later"""

    async def generate_text(self, **kwargs) -> str:
        """Generate text from the model."""
        raise NotImplementedError("Subclasses must implement generate_text")


@dataclass
class OllamaTextProvider(TextModelProvider):
    client: httpx.AsyncClient
    model_name: str

    async def generate_text(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 300,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate text using Ollama's API.

        Args:
            messages: List of message dictionaries (role, content)
            max_tokens: Maximum tokens to generate
            temperature: Temperature setting for generation
            **kwargs: Additional Ollama-specific parameters

        Returns:
            Generated text string
        """

        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
        )
        response.raise_for_status()
        response_data = response.json()

        return response_data["choices"][0]["message"]["content"]


@dataclass
class AudioProvider:
    repo_id: str = "sesame/csm-1b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
        max_audio_length_ms: int = 10_000,
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

        # Generate the audio using the CSM model
        audio = self.generator.generate(
            text=text,
            speaker=speaker_id,
            context=context_segments,
            max_audio_length_ms=max_audio_length_ms,
        )

        return audio

    def save_audio(self, audio, file_path):
        """
        Save the generated audio to a file.

        Args:
            audio: Audio tensor to save
            file_path: Path to save the audio file
        """
        torchaudio.save(file_path, audio.unsqueeze(0).cpu(), self.sample_rate)


class ConvoManager:
    """
    Manages a simulated conversation between multiple speakers using a language model.
    """

    def __init__(
        self,
        speakers: list[Speaker],
        text_provider: TextModelProvider,
        audio_provider: AudioProvider | None = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        audio_output_dir: str = "audio_output",
    ):
        """
        Initialize the conversation manager.

        Args:
            speakers: List of Speaker objects participating in the conversation
            text_provider: Model provider to use for text generation
            audio_provider: Optional AudioProvider for speech synthesis
            max_tokens: Maximum tokens to generate per turn
            temperature: Temperature setting for generation (higher = more random)
            system_prompt: Optional system prompt to guide the conversation
            audio_output_dir: Directory to save generated audio files
        """
        self.speakers = speakers
        self.text_provider = text_provider
        self.audio_provider = audio_provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.audio_output_dir = audio_output_dir

        # Default system prompt if none provided
        self.system_prompt = system_prompt or make_system_prompt(speakers)

        self.history: list[ConvoTurn] = []

    def _setup_audio_provider(self):
        if self.audio_output_dir:
            os.makedirs(self.audio_output_dir, exist_ok=True)

    def _audio_provider_turn_end(self, turn: ConvoTurn) -> ConvoTurn:
        if self.audio_provider and self.audio_output_dir:
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
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add previous conversation turns as context
        for turn in self.history:
            messages.append({"role": "assistant", "content": f"{turn.speaker.name}: {turn.text}"})

        # Add instruction for the next speaker
        if next_speaker:
            messages.append(
                {
                    "role": "user",
                    "content": f"Now {next_speaker.name} speaks next. Generate only {next_speaker.name}'s response.",
                }
            )

        return messages

    def select_next_speaker(self) -> Speaker:
        """
        Select the next speaker for the conversation.
        By default, rotates through speakers in order, avoiding consecutive turns.
        """
        if not self.history:
            return random.choice(self.speakers)

        last_speaker = self.history[-1].speaker
        available_speakers = [s for s in self.speakers if s != last_speaker]

        return random.choice(available_speakers)

    async def generate_turn(
        self, speaker: Speaker | None = None, generate_audio: bool = True
    ) -> ConvoTurn:
        """
        Generate the next turn in the conversation.

        Args:
            speaker: Optional speaker to force as the next speaker
            generate_audio: Whether to generate audio for this turn

        Returns:
            ConvoTurn object containing the generated text and optionally audio
        """
        speaker = speaker or self.select_next_speaker()
        messages = self._create_prompt_for_next_turn(speaker)

        # Generate text using the provider
        generated_text = await self.text_provider.generate_text(
            messages=messages, max_tokens=self.max_tokens, temperature=self.temperature
        )

        # Clean the response - remove speaker prefix if model included it
        if generated_text.startswith(f"{speaker.name}:"):
            generated_text = generated_text[len(f"{speaker.name}:") :].strip()

        # Create the turn with text
        turn = ConvoTurn(speaker=speaker, text=generated_text)
        self.history.append(turn)

        # Generate audio if requested and we have an audio generator
        if generate_audio and self.audio_provider:
            # Get recent context turns with audio (limit to 3 for performance)
            context_turns = [t for t in self.history[:-1] if t.audio is not None][-3:]

            # Generate audio
            audio = self.audio_provider.generate_audio(
                text=generated_text,
                speaker_id=speaker.speaker_id,
                context_turns=context_turns,
            )

            turn.audio = audio

            turn = self._audio_provider_turn_end(turn)

        return turn

    async def add_initial_phrase(
        self,
        initial_phrase: str,
        speaker: Speaker | None = None,
        generate_audio: bool = True,
    ) -> ConvoTurn:
        """
        Add an initial phrase to the conversation to seed the dialogue.

        Args:
            initial_phrase: The text to start the conversation with
            speaker: Optional speaker for the initial phrase (randomly selected if None)
            generate_audio: Whether to generate audio for this phrase

        Returns:
            The ConvoTurn object created for the initial phrase
        """
        if speaker is None:
            speaker = random.choice(self.speakers)

        # Create the turn with text
        turn = ConvoTurn(speaker=speaker, text=initial_phrase)
        self.history.append(turn)

        # Generate audio if requested and we have an audio generator
        if generate_audio and self.audio_provider:
            # Generate audio without context for the first turn
            audio = self.audio_provider.generate_audio(
                text=initial_phrase, speaker_id=speaker.speaker_id
            )

            # Update turn with audio
            turn.audio = audio

            turn = self._audio_provider_turn_end(turn)

        return turn

    async def generate_conversation_stream(
        self,
        num_turns: int,
        initial_phrase: str | None = None,
        initial_speaker: Speaker | None = None,
        generate_audio: bool = True,
    ):
        """
        Generate a conversation with the specified number of turns as an async generator.
        Yields each turn immediately after it's generated for real-time processing.

        Args:
            num_turns: Number of turns to generate
            initial_phrase: Optional text to start the conversation with
            initial_speaker: Optional speaker for the initial phrase
            generate_audio: Whether to generate audio for each turn

        Yields:
            ConvoTurn objects one at a time as they are generated
        """
        # Add initial phrase if provided
        if initial_phrase:
            turn = await self.add_initial_phrase(initial_phrase, initial_speaker, generate_audio)
            yield turn

        # Generate the rest of the conversation one turn at a time
        for _ in range(num_turns):
            turn = await self.generate_turn(generate_audio=generate_audio)
            yield turn

    async def generate_conversation(
        self,
        *args,
        **kwargs,
    ) -> list[ConvoTurn]:
        """
        Generate a conversation with the specified number of turns.

        Args:
            num_turns: Number of turns to generate
            initial_phrase: Optional text to start the conversation with
            initial_speaker: Optional speaker for the initial phrase
            generate_audio: Whether to generate audio for each turn

        Returns:
            List of ConvoTurn objects
        """
        return [turn async for turn in self.generate_conversation_stream(*args, **kwargs)]


# Example usage demonstration
async def demo():
    # Create speakers with speaker IDs for CSM model
    speakers = [
        Speaker(
            name="Alice",
            description="Tech entrepreneur",
            personality="Ambitious, technical, direct",
            speaking_style="Uses technical jargon, speaks confidently",
            speaker_id=0,  # Speaker ID for CSM
        ),
        Speaker(
            name="Bob",
            description="Philosophy professor",
            personality="Thoughtful, analytical, kind",
            speaking_style="Speaks in questions, uses metaphors",
            speaker_id=1,  # Speaker ID for CSM
        ),
        Speaker(
            name="Charlie",
            description="Stand-up comedian",
            personality="Witty, sarcastic, observant",
            speaking_style="Uses humor, makes pop culture references",
            speaker_id=0,  # Reusing speaker ID 0 since CSM has limited voices
        ),
    ]

    # Create the text provider (using Ollama by default)
    async with httpx.AsyncClient(base_url="http://localhost:11434/v1") as client:
        text_provider = OllamaTextProvider(client=client, model_name="gemma3:1b")

        # Create the audio generator (if you want audio)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        audio_provider = AudioProvider(device=device)

        # Initialize conversation manager with the providers
        manager = ConvoManager(
            speakers=speakers,
            text_provider=text_provider,
            audio_provider=audio_provider,
            audio_output_dir="audio_output",
        )

        # Example using the streaming generator
        print("Streaming conversation in real-time")
        print("-" * 50)

        initial_phrase = "Did you hear about that new AI model that just came out?"
        initial_speaker = speakers[0]  # Alice will start

        # Stream each turn as it's generated
        async for turn in manager.generate_conversation_stream(
            num_turns=5,
            initial_phrase=initial_phrase,
            initial_speaker=initial_speaker,
        ):
            # Process each turn immediately as it's generated
            print(f"{turn}")
            if turn.audio_path:
                print(f"Audio saved to: {turn.audio_path}")
            # You could add a delay here to simulate real-time conversation
            # await asyncio.sleep(1)

        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(demo())
