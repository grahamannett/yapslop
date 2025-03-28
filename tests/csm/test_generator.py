import os
import tempfile

import pytest
import torch
import torchaudio
from yapslop.generator import load_csm_1b, Generator


# --- Fixture for output directory using tempfile ---
@pytest.fixture
def audio_output_dir():
    """Fixture to create and cleanup a temporary audio output directory using tempfile."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def audio_output_file():
    """Fixture to create and cleanup a temporary audio output file using tempfile."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
        yield tmpfile.name


def test_generator_init(device):
    Generator.use_wm = True  # monkey patch the use_wm attribute to alter the output of load_csm_1b
    generator = load_csm_1b(device)
    assert generator._watermarker is not None, "Watermarker should be initialized if use_wm is True"


# --- Parameterized test function ---
@pytest.mark.parametrize("text", ["Hello from yapslop.", "Test audio generation."])
def test_audio_generation(audio_output_file: str, text: str, device: str):
    """
    Test audio generation with different text prompts using CSM model.

    This test:
    - Downloads the CSM-1B model.
    - Generates audio for a given text prompt.
    - Saves the generated audio to a temporary directory created by tempfile.
    - Asserts that the audio file is created, has the correct sample rate,
      is mono, and is not empty.
    - The temporary directory and its contents are automatically cleaned up after the test.
    """
    assert audio_output_file.endswith(".wav")
    generator = load_csm_1b(device)

    audio = generator.generate(text=text, speaker=0, context=[], max_audio_length_ms=10_000)
    torchaudio.save(audio_output_file, audio.unsqueeze(0).cpu(), generator.sample_rate)

    # for integration tests, prefer the saved audio file versus the audio tensor
    assert os.path.exists(audio_output_file)
    waveform, sample_rate = torchaudio.load(audio_output_file)
    assert sample_rate == generator.sample_rate
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0


@pytest.mark.parametrize("text", ["We are streaming from yapslop", "This is a streamed "])
def test_stream_audio_generation(text: str) -> None:
    """
    Test streaming audio generation with different text prompts using CSM model.

    This test:

    """
    pass


@pytest.mark.parametrize("text,speaker_id", [("Testing the tokenizer", 0)])
def test_tokenize_text(text, speaker_id, generator):
    text_tokens, text_masks = generator._tokenize_text_segment(text, speaker_id)

    assert text_tokens.shape[1] == 33
    assert text_masks.shape[1] == 33
    assert text_tokens.dtype == torch.int64
    assert text_masks.dtype == torch.bool
