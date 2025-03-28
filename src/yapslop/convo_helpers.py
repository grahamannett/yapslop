import json
import re
from functools import cache, partial
from typing import Any, Callable, Sequence, TypeAlias

from yapslop.convo_dto import ConvoTurn, Speaker
from yapslop.providers.parsers import remove_thinking_block
from yapslop.utils.autils import allow_retry
from yapslop.utils.schema_generator import get_type_schema

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


def parse_json_content(content: str) -> dict:
    """
    Parse JSON content from model response, handling various formats.

    While using tool calling would be better, not all the models support it and seems like some of the models I am using are very
    inconsistent with format, e.g. using `" " instead of `"' or not surrounding field with `"`

    Args:
        content: Raw response from model as string or dict

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If content is not a string or dict
        JSONDecodeError: If content cannot be parsed as valid JSON
    """

    content = remove_thinking_block(content)

    # check if the content has a ```json block which we should extract
    if "```" in content:
        content_match = re.search(r"```(?:json)?\n(.*?)\n```", content, re.DOTALL)
        content = content_match.group(1) if content_match else content

    try:
        return json.loads(content)  # type: ignore
    except json.JSONDecodeError as err:
        raise err


async def tool_generate_speaker(text_provider, speakers: list[Speaker] | None = None) -> dict:
    """
    Generate a new character for a conversation using a language model.
    """
    # messages = make_messages(gen_speaker_prompt, system=gen_speaker_system_prompt)
    speaker_names = ""
    if speakers:
        speaker_names = f"Speakers so far: {', '.join(s.name for s in speakers)}"

    prompt = gen_speaker_prompt.format(speaker_names=speaker_names, gen_speaker_example=gen_speaker_example)
    messages = make_messages(prompt, system=gen_speaker_system_prompt)
    speaker_schema = get_type_schema(Speaker)
    # resp = await text_provider.chat_oai(messages, model_options={"tools": [speaker_schema]})
    resp = await text_provider.chat_ollama(messages, tools=[speaker_schema])

    if text_provider._parser:
        thinking, content = text_provider._parser.extract(resp)
        resp = content[0]

    resp = parse_json_content(resp)

    return resp


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
        raise ValueError(f"Failed to convert content to Speaker: {err}") from err


# i like this better than decorating, but as noted still an awkward pattern b/c cant pass by name here unless also
# pass by name in the caller of the partial
generate_speaker_allow_retry = partial(allow_retry, generate_speaker)
