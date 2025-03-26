import torch
import torchaudio

from yapslop.convo_dto import ConvoTurn


def load_audio(audio_path: str, new_freq: int = 24_000) -> torch.Tensor:
    """
    Load an audio file and resample it to the specified frequency.

    Args:
        audio_path: Path to the audio file to load
        new_freq: Target sample rate to resample to (default: 24kHz)

    Returns:
        Resampled audio tensor
    """
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    return torchaudio.functional.resample(audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=new_freq)


def save_combined_audio(
    output_file: str,
    turns: list[ConvoTurn],
    add_silence_ms: int = 500,
    audio_provider=None,
) -> None:
    """
    Combine audio from multiple conversation turns into a single audio file with silence between each turn.

    Args:
        output_file: Path where the combined audio file will be saved
        turns: Optional list of ConvoTurn objects containing audio to combine. If None, uses self.history
        add_silence_ms: Milliseconds of silence to insert between each turn. Defaults to 500ms

    Returns:
        str: Path to the saved combined audio file

    Raises:
        ValueError: If no audio provider is initialized or if no turns with audio are found
    """
    if audio_provider is None:
        raise ValueError("Audio provider not initialized")

    # Use provided turns or conversation history

    # Extract audio tensors from turns that have audio
    audio_tensors = [turn.audio for turn in turns if turn.audio is not None]

    if not audio_tensors:
        raise ValueError("No turns with audio found")

    # Calculate silence samples
    silence_samples = int(audio_provider.sample_rate * add_silence_ms / 1000)
    silence = torch.zeros(silence_samples, device=audio_provider.device)

    # Combine audio tensors with silence between them
    combined_tensors = []
    for i, audio in enumerate(audio_tensors):
        combined_tensors.append(audio)
        # Add silence after each segment except the last
        if i < len(audio_tensors) - 1:
            combined_tensors.append(silence)

    # Concatenate all audio tensors along the time dimension
    final_audio = torch.cat(combined_tensors)

    # Save the combined audio
    torchaudio.save(
        output_file,
        final_audio.unsqueeze(0).cpu(),  # Add channel dimension and ensure on CPU
        audio_provider.sample_rate,
    )
