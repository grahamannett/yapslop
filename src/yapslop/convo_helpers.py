import json
import re
from typing import Sequence, TypeAlias, Callable, Any
from yapslop.convo_dto import ConvoTurn, Speaker

MessageType: TypeAlias = list[dict[str, str]]

gen_speaker_system_prompt = """You are part of an AI system that helps to create characters for a conversation simulation.
Output ONLY the JSON object and do not include additional information or text."""

gen_speaker_prompt = """{speaker_names}
Generate a character that has the following properties:
- name: The first name of the character.
- description: Description of the character containing the description of the character and speaking style
e.g. 'Tech entrepreneur who speaks in technical jargon, speaks confidently'
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


async def generate_speaker(chat_func: Callable[..., Any], speakers: list[Speaker] | None = None) -> Speaker:
    """
    Generate a new speaker character using a language model.

    Args:
        chat_func: Async function that interfaces with the language model
        speakers: Optional list of existing speakers to avoid duplication

    Returns:
        A Speaker object with generated name and description

    Raises:
        ValueError: If speaker generation or JSON parsing fails
    """
    speakers = speakers or []

    speaker_names = f"Speakers so far: {', '.join(s.name for s in speakers)}" if speakers else ""

    prompt = gen_speaker_prompt.format(speaker_names=speaker_names)
    messages = make_messages(prompt, system=gen_speaker_system_prompt)

    try:
        response = await chat_func(messages=messages, stream=False)
        return Speaker(**parse_json_content(response))
    except Exception as e:
        raise ValueError(f"Failed to generate speaker: {e}") from e


def parse_json_content(content: str | dict) -> dict:
    """
    Parse JSON content from model response, handling various formats.

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

    # check if the content has a ```json block which we should extract
    if "```" in content:
        match = re.search(r"```(?:json)?\n(.*?)\n```", content, re.DOTALL)
        content = match.group(1) if match else content

    try:
        return json.loads(content)  # type: ignore
    except json.JSONDecodeError:
        raise json.JSONDecodeError("Invalid JSON content", doc=str(content), pos=0)
