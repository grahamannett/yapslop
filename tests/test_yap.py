from os import getenv
import httpx
import pytest
from json import JSONDecodeError
from textwrap import indent

from yapslop.convo_helpers import generate_speaker, generate_speaker_allow_retry, _generate_speaker_resp
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
def ollama_client_params():
    return {
        "base_url": getenv("ollama_url", "http://localhost:11434"),
        "timeout": int(getenv("ollama_timeout", 10)),
    }


@pytest.fixture
def ollama_model_params():
    return {
        "model_name": getenv("ollama_model", "gemma3:latest"),
    }


@pytest.fixture
def ollama_reasoning_model_params():
    return {
        "model_name": getenv("ollama_reasoning_model", "qwq:latest"),
    }


@pytest.fixture
def ollama_params(ollama_client_params, ollama_model_params):
    return {"client_kwargs": ollama_client_params, "ollama_kwargs": ollama_model_params}


@pytest.fixture
def ollama_reasoning_params(ollama_client_params, ollama_reasoning_model_params):
    return {"client_kwargs": ollama_client_params, "ollama_kwargs": ollama_reasoning_model_params}


@pytest.fixture
async def text_provider(ollama_params: dict):
    async with httpx.AsyncClient(**ollama_params["client_kwargs"]) as client:
        yield TextProvider(client, **ollama_params["ollama_kwargs"])


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


async def test_convo_generate_speakers(text_provider: TextProvider):
    """
    Test that the convo manager can generate speakers
    """
    convo_manager = ConvoManager(text_provider, n_speakers=2)
    await convo_manager.setup_speakers()

    assert len(convo_manager.speakers) == 2
