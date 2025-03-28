from os import getenv

import httpx
import pytest
from torch import cuda

from yapslop.convo_dto import Speaker

from yapslop.providers.yaproviders import TextProvider

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def device():
    return getenv("DEVICE", "cuda" if cuda.is_available() else "cpu")


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


@pytest.fixture
def generator(device):
    from yapslop.generator import load_csm_1b

    return load_csm_1b(device)


@pytest.fixture
def csm_generator(device):
    """
    Fixture to load the CSM-1B model, mostly for benchmarking or verifying yapslop.generator

    Note that using the csm.generator.load_csm_1b will raise multiple warnings related to loading the watermarker
    this is expected and is from silentcipher.server.Model since torch.load is used to load the checkpoints
    tbh it seems like the load_models in that is just loading state_dicts, so probably can be fixed if just passing
    torch.load(..., weights_only=True), dunno exactly though but think its this:
    - https://github.com/sony/silentcipher/blob/c11afe0e2df73f1ac0f848e2903b8af36e6a0b10/src/silentcipher/server.py#L444
    """
    from csm.generator import load_csm_1b

    return load_csm_1b(device)
