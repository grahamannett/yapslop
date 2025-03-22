from pathlib import Path

import httpx
import asyncio

from yapslop.yap import ConvoManager, HTTPConfig, ProvidersSetup
from yapslop.yap_common import initial_speakers
from yapslop.audio_helpers import save_combined_audio

_line_sep = "-" * 50


def _setup_demo(audio_output_dir: str, cleanup_audio_dir: bool):
    """
    Set up the audio output directory for the demo.

    Args:
        audio_output_dir: Directory path where audio files will be saved
        cleanup_audio_dir: If True, removes any existing WAV files in the directory
    """
    if cleanup_audio_dir and (_dir := Path(audio_output_dir)):
        _dir.mkdir(parents=True, exist_ok=True)
        _ = [f.unlink() for f in _dir.glob("*.wav")]


# Example usage demonstration
async def demo(
    n_speakers: int = 3,
    num_turns: int = 5,
    initial_text: str = "Did you hear about that new conversational AI model that just came out?",
    audio_output_dir: str = "audio_output",
    combined_audio_file: str = "combined.wav",
    cleanup_audio_dir: bool = True,
    max_audio_length_ms: int = 90_000,
):
    """
    Demonstrate a multi-speaker conversation with audio generation.

    This function sets up a conversation with one initial speaker and generates
    additional speakers as needed. It streams the conversation turn by turn,
    generating both text and audio for each turn.

    Args:
        n_speakers: Total number of speakers to include in the conversation
        num_turns: Number of conversation turns to generate
        initial_text: The starting text for the conversation
        audio_output_dir: Directory where individual audio files will be saved
        combined_audio_file: Filename for the combined audio of all turns
        cleanup_audio_dir: If True, cleans up existing audio files before starting
        max_audio_length_ms: Maximum length of generated audio in milliseconds
    """

    _setup_demo(audio_output_dir, cleanup_audio_dir)

    async with httpx.AsyncClient(base_url=HTTPConfig.base_url) as client:
        text_provider, audio_provider = ProvidersSetup(
            configs={
                "text": {"client": client},
                "audio": {},
            }
        )

        convo_manager = ConvoManager(
            n_speakers=n_speakers,
            speakers=initial_speakers,
            text_provider=text_provider,
            audio_provider=audio_provider,
            audio_output_dir=audio_output_dir,
        )

        await convo_manager.setup_speakers()

        print("StreamSpeakers", [s.name for s in convo_manager.speakers])
        print(_line_sep)

        async for turn in convo_manager.generate_convo_stream(
            num_turns=num_turns,
            initial_text=initial_text,
            initial_speaker=convo_manager.speakers[0],
            max_audio_length_ms=max_audio_length_ms,
        ):
            print(f">>{turn}")
            if turn.audio_path:
                print(f"\tAudio saved to: {turn.audio_path}")

        print(_line_sep)

        save_combined_audio(
            output_file=combined_audio_file,
            turns=convo_manager.history,
            audio_provider=convo_manager.audio_provider,
        )
        print(f"Combined audio saved to: {combined_audio_file}")


if __name__ == "__main__":
    asyncio.run(demo())
