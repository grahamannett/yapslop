from pathlib import Path

import httpx
import asyncio

from yapslop.yap import ConvoManager, HTTPConfig, ProvidersSetup
from yapslop.yap_common import initial_speakers

_line_sep = "-" * 50


def _setup_demo(audio_output_dir: str, cleanup_audio_dir: bool):
    if cleanup_audio_dir and (_dir := Path(audio_output_dir)):
        _dir.mkdir(parents=True, exist_ok=True)
        _ = [f.unlink() for f in _dir.glob("*.wav")]


# Example usage demonstration
async def demo(
    n_speakers: int = 3,
    num_turns: int = 5,
    initial_text: str = "Did you hear about that new conversational AI model that just came out?",
    audio_output_dir: str = "audio_output",
    cleanup_audio_dir: bool = True,
    max_audio_length_ms: int = 90_000,
):
    """
    demo showing one initial speaker and then generate the rest of the speakers
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

        initial_speaker = convo_manager.speakers[0]
        print("StreamConvo with Speakers", [s.name for s in convo_manager.speakers])
        print(_line_sep)

        async for turn in convo_manager.generate_convo_stream(
            num_turns=num_turns,
            initial_text=initial_text,
            initial_speaker=initial_speaker,
            max_audio_length_ms=max_audio_length_ms,
        ):
            print(f">>{turn}")
            if turn.audio_path:
                print(f"\tAudio saved to: {turn.audio_path}")

        print(_line_sep)

    # Save combined audio
    combined_audio_path = convo_manager.save_combined_audio(output_path="full_conversation.wav")
    print(f"Combined audio saved to: {combined_audio_path}")


if __name__ == "__main__":
    asyncio.run(demo())
