from pathlib import Path

import httpx
import asyncio

from yapslop.yap import AudioProvider, ConvoManager, Speaker, TextProvider


def _setup_demo(audio_output_dir: str, cleanup_audio_dir: bool):
    if cleanup_audio_dir and (_dir := Path(audio_output_dir)):
        _dir.mkdir(parents=True, exist_ok=True)
        _ = [f.unlink() for f in _dir.glob("*.wav")]


# Example usage demonstration
async def demo(
    n_speakers: int = 2,
    num_turns: int = 5,
    initial_phrase: str = "Did you hear about that new conversational AI model that just came out?",
    audio_output_dir: str = "audio_output",
    cleanup_audio_dir: bool = True,
):
    _setup_demo(audio_output_dir, cleanup_audio_dir)

    # demo showing one initial speaker but then generate the rest of the speakers
    speakers = [
        Speaker(
            name="Seraphina",
            description="Tech entrepreneur. Uses technical jargon, speaks confidently",
        ),
    ]

    # Create the text provider (using Ollama by default). Example using the streaming generator
    audio_provider = AudioProvider()
    text_provider = TextProvider()

    async with httpx.AsyncClient(base_url=text_provider.base_url) as client:
        text_provider.client = client

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
