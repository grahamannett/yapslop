import pytest


@pytest.fixture
def benchmark_input():
    return "This is a benchmark test of the tokenization speed", 0


@pytest.mark.benchmark(group="tokenize_text", min_rounds=5)
def test_benchmark_tokenize_text(generator, benchmark, benchmark_input):
    """Benchmark text tokenization performance between Generator implementations"""
    text, speaker_id = benchmark_input

    def run_tokenize():
        return generator._tokenize_text_segment(*benchmark_input)

    benchmark(run_tokenize)


@pytest.mark.benchmark(group="tokenize_text", min_rounds=5)
def test_benchmark_csm_tokenize_text(csm_generator, benchmark, benchmark_input):
    """Benchmark CSM text tokenization for comparison"""

    def run_tokenize():
        return csm_generator._tokenize_text_segment(*benchmark_input)

    benchmark(run_tokenize)


"""
TODO: find why there is warning for csm load_csm_1b and then remove below
"""


def test_load_csm():
    from csm.generator import load_csm_1b

    csm_generator = load_csm_1b()
    print("got csm generator")


def test_load_yapslop():
    from yapslop.generator import load_csm_1b

    yapslop_generator = load_csm_1b()
    print("got yapslop generator")
