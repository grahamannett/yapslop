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


def msg_helper(*msgs: Sequence[str] | str, system: str | None = None) -> MessageType:
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for i, m in enumerate(msgs):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": m})
    return out


def get_conversation_as_string(history: list[ConvoTurn]) -> str:  # type: ignore
    return "\n".join([str(turn) for turn in history])


async def generate_speaker(
    chat_func: Callable[..., Any], speakers: list[Speaker] | None = None
) -> Speaker:
    """
    Generate a new speaker character using a language model.

    Args:
        text_provider: The text provider to use for generating the speaker
        speakers: Optional list of existing speakers to avoid duplication

    Returns:
        A dictionary containing the speaker properties
    """
    speakers = speakers or []

    speaker_names = f"Speakers so far: {', '.join(s.name for s in speakers)}" if speakers else ""

    prompt = gen_speaker_prompt.format(speaker_names=speaker_names)
    messages = msg_helper(prompt, system=gen_speaker_system_prompt)

    try:
        response = await chat_func(messages=messages, stream=False)
        return Speaker(**parse_json_content(response))
    except Exception as e:
        raise ValueError(f"Failed to generate speaker: {e}") from e


def parse_json_content(content: str | dict) -> dict:
    """
    since tool use can be iffy with the models I have been using, its worth trying to just ask the
    model to explicitly output the json info
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
