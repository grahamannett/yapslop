import time
import pytest
import torch


@pytest.fixture
def benchmark_input():
    return "First benchmark sentence", 0


@pytest.fixture
def benchmark_input_b():
    return "Second benchmark sentence but longer", 1


@pytest.fixture
def benchmark_input_batch(benchmark_input, benchmark_input_b):
    return list(zip(benchmark_input, benchmark_input_b))
    # texts, speaker_ids = zip(*[benchmark_input, benchmark_input_b])
    # return list(texts), list(speaker_ids)


@pytest.mark.benchmark(group="tokenize_text", min_rounds=5, timer=time.time, disable_gc=True, warmup=False)
def test_benchmark_tokenize_text(generator, benchmark, benchmark_input):
    """Benchmark text tokenization performance between Generator implementations"""
    text, speaker_id = benchmark_input

    def run_tokenize():
        return generator._tokenize_text_segment(*benchmark_input)

    benchmark(run_tokenize)


@pytest.mark.benchmark(group="tokenize_text", min_rounds=5, timer=time.time, disable_gc=True, warmup=False)
def test_benchmark_csm_tokenize_text(csm_generator, benchmark, benchmark_input):
    """Benchmark CSM text tokenization for comparison"""

    def run_tokenize():
        return csm_generator._tokenize_text_segment(*benchmark_input)

    benchmark(run_tokenize)


@pytest.mark.benchmark(group="tokenize_text", min_rounds=5, timer=time.time, disable_gc=True, warmup=False)
def test_benchmark_tokenize_text_fast(generator, benchmark, benchmark_input):
    """Benchmark CSM text tokenization for comparison"""

    benchmark(generator._tokenize_text_segment_fast, *benchmark_input)


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


def test_tokenize_text(generator, csm_generator, benchmark_input):
    """Test that the fast tokenization method is equivalent to the slow method"""
    text, speaker_id = benchmark_input

    tokens, mask = generator._tokenize_text_segment_fast(text, speaker_id)
    csm_tokens, csm_mask = csm_generator._tokenize_text_segment(text, speaker_id)

    assert torch.allclose(tokens, csm_tokens)
    assert torch.allclose(mask, csm_mask)
