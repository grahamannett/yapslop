import pytest
from json import JSONDecodeError
from textwrap import indent

from yapslop.convo_helpers import (
    generate_speaker,
    generate_speaker_allow_retry,
    _generate_speaker_resp,
    tool_generate_speaker,
)
from yapslop.yap import ConvoManager, Speaker
from yapslop.providers.yaproviders import TextProvider
from yapslop.yap_common import initial_speakers

pytestmark = pytest.mark.anyio


def test_speakers_from_dict(speakers_dict: list[dict]) -> None:
    speakers = Speaker.from_data(speakers_dict)

    assert len(speakers) == len(speakers_dict)
    for speaker, speaker_dict in zip(speakers, speakers_dict):
        assert speaker.name == speaker_dict["name"]
        assert speaker.description == speaker_dict["description"]


def test_initial_speakers() -> None:
    assert isinstance(initial_speakers[0], dict)
    assert "name" in initial_speakers[0] and "description" in initial_speakers[0]

    speakers = Speaker.from_data(initial_speakers)

    assert isinstance(speakers[0], Speaker)
    assert speakers[0].name == initial_speakers[0]["name"]
    assert speakers[0].description == initial_speakers[0]["description"]


async def test_generate_speaker(text_provider: TextProvider):
    """Test speaker generation (requires configured API server)."""
    resp, prompt = await _generate_speaker_resp(gen_func=text_provider)
    assert isinstance(resp, str) and "name" in resp

    # without retry, might error based on model
    try:
        speaker = await generate_speaker(gen_func=text_provider)
        assert isinstance(speaker, Speaker)
    except JSONDecodeError as err:
        print(err)
        print(indent(err.doc, "|"))


async def test_generate_speaker_allow_retry(text_provider: TextProvider):
    """
    Test that the generate_speaker_kwargs function can generate a speaker dictionary
    """
    speaker_dict: dict = await generate_speaker_allow_retry(text_provider, to_type=lambda x: x)
    assert "name" in speaker_dict and "description" in speaker_dict

    speaker = Speaker(**speaker_dict)

    assert speaker.name == speaker_dict["name"]
    assert speaker.description == speaker_dict["description"]

    speaker = await generate_speaker_allow_retry(text_provider, max_retries=5)
    assert isinstance(speaker, Speaker)
    assert hasattr(speaker, "name") and hasattr(speaker, "description")


async def test_tool_generate_speaker(text_provider: TextProvider) -> None:
    """Test tool-based speaker generation (requires Ollama server with tool calling)."""
    speaker_resp = await tool_generate_speaker(text_provider)


async def test_convo_generate_speakers(text_provider: TextProvider):
    """
    Test that the convo manager can generate speakers
    """
    convo_manager = ConvoManager(text_provider, n_speakers=2)
    await convo_manager.setup_speakers()

    assert len(convo_manager.speakers) == 2
