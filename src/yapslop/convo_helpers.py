import json
import re

from functools import cache, partial
from typing import Sequence, TypeAlias, Callable, Any
from yapslop.convo_dto import ConvoTurn, Speaker
from yapslop.utils.autils import allow_retry

MessageType: TypeAlias = list[dict[str, str]]


gen_speaker_system_prompt = (
    "You are part of an AI system that helps to create characters for a conversation simulation."
)

gen_speaker_example = """
An example is:
```json
{
    "name": "Bob",
    "description": "Tech entrepreneur who speaks in technical jargon, speaks confidently"
}
```"""

gen_speaker_prompt = """{speaker_names}
Generate a character that has the following properties:
- name: The first name of the character.
- description: Description of the character containing the description of the character and speaking style

{gen_speaker_example}

Output ONLY the JSON object and do not include additional information or text.
"""


def _msg_role(i: int) -> str:
    """
    Determine message role based on index position.

    Args:
        i: Index position in message sequence

    Returns:
        "user" for even indices, "assistant" for odd indices
    """
    return "user" if i % 2 == 0 else "assistant"


@cache
def make_messages(*msgs: Sequence[str] | str, system: str | None = None) -> MessageType:
    """
    Create a list of message dictionaries for chat API input.

    Args:
        *msgs: Variable number of message strings or sequences
        system: Optional system prompt to include at start

    Returns:
        List of message dictionaries with role and content
    """
    out = []

    if system:
        out += [{"role": "system", "content": system}]

    out += [{"role": _msg_role(i), "content": m} for i, m in enumerate(msgs)]
    return out


def get_conversation_as_string(history: list[ConvoTurn]) -> str:  # type: ignore
    """
    Convert conversation history to a string representation.

    Args:
        history: List of conversation turns

    Returns:
        String containing all turns concatenated with newlines
    """
    return "\n".join([str(turn) for turn in history])


def parse_json_content(content: str | dict) -> dict:
    """
    Parse JSON content from model response, handling various formats.

    While using tool calling would be better, not all the models support it and seems like some of the models I am using are very
    inconsistent with format, e.g. using `“` instead of `"' or not surrounding field with `"`

    Args:
        content: Raw response from model as string or dict

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If content is not a string or dict
        JSONDecodeError: If content cannot be parsed as valid JSON
    """
    # if pass in the raw response, extract the content
    if isinstance(content, dict):
        content = content.get("message", content).get("content", content)

    if not isinstance(content, str):
        raise ValueError(f"Expected string content, got {type(content)}: {content}")

    # if parsing json explicitly, remove thinking tags, should keep this info if not just generating characters
    if all(x in content for x in ["<think>", "</think>"]):
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # check if the content has a ```json block which we should extract
    if "```" in content:
        match = re.search(r"```(?:json)?\n(.*?)\n```", content, re.DOTALL)
        content = match.group(1) if match else content

    try:
        return json.loads(content)  # type: ignore
    except json.JSONDecodeError as err:
        raise err


async def _generate_speaker_resp(
    gen_func: Callable[..., Any],
    speakers: list[Speaker] | None = None,
) -> tuple[Any, str]:
    """
    Generate a new speaker character using a language model.
    Only handles the API request, not the response parsing.

    Args:
        gen_func: Async function that interfaces with the language model
        speakers: Optional list of existing speakers to avoid duplication

    Returns:
        Raw response from the chat function
    """
    speakers = speakers or []

    # Prepare messages
    speaker_names = ""
    if speakers:
        speaker_names = f"Speakers so far: {', '.join(s.name for s in speakers)}"

    prompt = gen_speaker_prompt.format(speaker_names=speaker_names, gen_speaker_example=gen_speaker_example)
    resp = await gen_func(make_messages(prompt, system=gen_speaker_system_prompt))
    return resp, prompt


async def generate_speaker(
    gen_func: Callable[..., Any],
    speakers: list[Speaker] | None = None,
    to_type: Callable[[dict], Any] = lambda x: Speaker(**x),
):
    """
    Generate a new speaker character using a language model.
    This is split out to allow testing of JUST the response
    """
    content, prompt = await _generate_speaker_resp(gen_func, speakers)
    try:
        parsed_resp = parse_json_content(content)
        return to_type(parsed_resp)
    except json.JSONDecodeError as err:
        raise ValueError(f"Failed to parse JSON content: {err}") from err
    except Exception as err:
        raise ValueError(f"Failed to convert conten to Speaker: {err}") from err


# i like this better than decorating, but as noted still an awkward pattern b/c cant pass by name here unless also
# pass by name in the caller of the partial
generate_speaker_allow_retry = partial(allow_retry, generate_speaker)
