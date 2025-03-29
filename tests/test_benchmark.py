import time
import pytest
import torch


from yapslop.convo_dto import BaseConvoTurn, Segment, _load_segment_audio


@pytest.fixture
def benchmark_input():
    return "First benchmark sentence", 0


@pytest.fixture
def benchmark_input_b():
    return "Second benchmark sentence but longer", 1


@pytest.fixture
def benchmark_input_batch(benchmark_input, benchmark_input_b):
    return list(zip(benchmark_input, benchmark_input_b))


@pytest.fixture
def benchmark_convo_turn():
    return BaseConvoTurn.from_audio_path(
        "./audio_output/turn_0_speaker_0.wav",
        "Did you hear about that new conversational AI model that just came out?",
    )


@pytest.fixture
def benchmark_convo_turn_b():
    return BaseConvoTurn.from_audio_path(
        "./audio_output/turn_1_speaker_1.wav",
        "I dont know the audio for this tbh",
    )


@pytest.fixture
def benchmark_convo_turn_batch(benchmark_convo_turn, benchmark_convo_turn_b):
    return [benchmark_convo_turn, benchmark_convo_turn_b]


@pytest.mark.benchmark(group="tokenize_text", min_rounds=5, timer=time.time, disable_gc=True, warmup=False)
def test_benchmark_tokenize_text(generator, benchmark, benchmark_input):
    """Benchmark text tokenization performance between Generator implementations"""
    text, speaker_id = benchmark_input

    benchmark(generator._tokenize_text_segment, *benchmark_input)


@pytest.mark.benchmark(group="tokenize_text", min_rounds=5, timer=time.time, disable_gc=True, warmup=False)
def test_benchmark_csm_tokenize_text(csm_generator, benchmark, benchmark_input):
    """Benchmark CSM text tokenization for comparison"""

    benchmark(csm_generator._tokenize_text_segment, *benchmark_input)


@pytest.mark.benchmark(group="tokenize_text", min_rounds=5, timer=time.time, disable_gc=True, warmup=False)
def test_benchmark_tokenize_text_fast(generator, benchmark, benchmark_input):
    """Benchmark CSM text tokenization for comparison"""

    benchmark(generator._tokenize_text_segment_fast, *benchmark_input)


@pytest.mark.benchmark(group="tokenize_audio", min_rounds=5, timer=time.time, disable_gc=True, warmup=False)
def test_benchmark_tokenize_audio(generator, benchmark, benchmark_convo_turn):
    """Benchmark the slow audio tokenization method."""
    benchmark(generator._tokenize_audio, benchmark_convo_turn.audio)


@pytest.mark.benchmark(group="tokenize_audio", min_rounds=5, timer=time.time, disable_gc=True, warmup=False)
def test_benchmark_tokenize_audio_fast(generator, benchmark, benchmark_convo_turn):
    """Benchmark the fast audio tokenization method."""
    benchmark(generator._tokenize_audio_fast, benchmark_convo_turn.audio)


def test_tokenize_fast(generator, benchmark_input):
    """Test that the fast tokenization method is equivalent to the slow method"""
    text, speaker_id = benchmark_input

    slow_tokens, slow_masks = generator._tokenize_text_segment(text, speaker_id)
    fast_tokens, fast_masks = generator._tokenize_text_segment_fast(text, speaker_id)

    assert torch.allclose(slow_tokens, fast_tokens)
    assert torch.allclose(slow_masks, fast_masks)


def test_tokenize_fast_batch(generator, benchmark_input_batch):
    texts, speaker = benchmark_input_batch

    batch_toks, batch_mask = generator._tokenize_text_segment_fast(texts, speaker)

    toks_a, mask_a = generator._tokenize_text_segment(texts[0], speaker[0])
    toks_b, mask_b = generator._tokenize_text_segment(texts[1], speaker[1])

    assert batch_toks[0].sum() == toks_a.sum()
    assert batch_toks[1].sum() == toks_b.sum()

    # avoiding having to worry about the diferent tensor sizes but i know the second seq is longer,
    assert (batch_mask[0, : mask_a.shape[-2], : mask_a.shape[-1]] == mask_a).all()
    assert (batch_mask[1] == mask_b).all()


def test_tokenize_text(generator, csm_generator, benchmark_input):
    """Test that the fast tokenization method is equivalent to the slow method"""
    text, speaker_id = benchmark_input

    tokens, mask = generator._tokenize_text_segment_fast(text, speaker_id)
    csm_tokens, csm_mask = csm_generator._tokenize_text_segment(text, speaker_id)

    assert torch.allclose(tokens, csm_tokens)
    assert torch.allclose(mask, csm_mask)


def test_tokenize_audio(generator, benchmark_convo_turn):
    fast_tokens, fast_mask = generator._tokenize_audio_fast(benchmark_convo_turn.audio)
    slow_tokens, slow_mask = generator._tokenize_audio(benchmark_convo_turn.audio)

    assert torch.allclose(fast_tokens, slow_tokens)
    assert torch.allclose(fast_mask, slow_mask)


def test_combine_audio(generator, benchmark_convo_turn_batch):
    audio_tensors = [turn.audio for turn in benchmark_convo_turn_batch]
    print(f"tensor sizes: {[t.shape for t in audio_tensors]}")

    combined_audio = generator._combine_audio_tensors(*audio_tensors)

    toks, mask = generator._tokenize_audio_fast(combined_audio)
