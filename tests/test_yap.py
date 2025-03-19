import httpx
import pytest

from yapslop.convo_helpers import generate_speaker_dict
from yapslop.yap import Speaker, TextProvider, ConvoManager

pytestmark = pytest.mark.anyio


@pytest.fixture
def speakers():
    """
    Speakers for the conversation.
    Use this fixture to avoid having to create them and avoiding doing a mock of the speaker response
    """
    return [
        Speaker(
            name="Alice",
            description="Tech entrepreneur. Uses technical jargon, speaks confidently",
        ),
        Speaker(
            name="Bob",
            description="Philosophy professor. Speaks in questions, uses metaphors",
        ),
        Speaker(
            name="Charlie",
            description="Stand-up comedian. Uses humor, makes pop culture references",
        ),
    ]


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


async def test_generate_speaker(text_provider: TextProvider):
    """
    Test that the generate_speaker_dict function can generate a speaker dictionary
    """
    speaker_dict = await generate_speaker_dict(text_provider)

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
