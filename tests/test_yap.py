import httpx
import pytest

from yapslop.convo_helpers import generate_speaker
from yapslop.yap import ConvoManager, Speaker, TextProvider
from yapslop.yap_common import initial_speakers

pytestmark = pytest.mark.anyio


@pytest.fixture
def speakers_dict():
    """
    Dictionary of speakers for the conversation.
    Use this fixture to avoid having to create them and avoiding doing a mock of the speaker response
    """
    return [
        {
            "name": "Alice",
            "description": "Tech entrepreneur. Uses technical jargon, speaks confidently",
        },
        {
            "name": "Bob",
            "description": "Philosophy professor. Speaks in questions, uses metaphors",
        },
        {
            "name": "Charlie",
            "description": "Stand-up comedian. Uses humor, makes pop culture references",
        },
    ]


@pytest.fixture
def speakers(speakers_dict):
    """
    Speaker objects for the conversation.
    Created from the speakers_dict fixture.
    """
    return Speaker.from_data(speakers_dict)


@pytest.fixture
def ollama_params():
    return {
        "ollama_url": "http://localhost:11434",
        "model_name": "gemma3:latest",
    }


@pytest.fixture
async def text_provider(ollama_params: dict):
    base_url = ollama_params.pop("ollama_url")
    async with httpx.AsyncClient(base_url=base_url) as client:
        yield TextProvider(client, **ollama_params)


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
    """
    Test that the generate_speaker_kwargs function can generate a speaker dictionary
    """
    speaker_dict = await generate_speaker(text_provider)

    assert "name" in speaker_dict and "description" in speaker_dict

    speaker = Speaker(**speaker_dict)

    assert speaker.name == speaker_dict["name"]
    assert speaker.description == speaker_dict["description"]


async def test_convo_generate_speakers(text_provider: TextProvider):
    """
    Test that the convo manager can generate speakers
    """
    convo_manager = ConvoManager(text_provider, n_speakers=2)
    await convo_manager.setup_speakers()

    assert len(convo_manager.speakers) == 2
