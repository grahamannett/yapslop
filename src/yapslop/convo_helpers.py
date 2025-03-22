import json
import re
from typing import Sequence, TypeAlias, TYPE_CHECKING

if TYPE_CHECKING:  # to avoid circular imports since initially was trying to utilize 1 file
    from yapslop.yap import ConvoTurn, TextProvider, Speaker

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


def get_conversation_as_string(history: list["ConvoTurn"]) -> str:  # type: ignore
    return "\n".join([str(turn) for turn in history])


async def generate_speaker_dict(text_provider: "TextProvider", speakers: list["Speaker"] = []):
    # create the list of speaker names for the prompt, possible to put on either system prompt or user prompt
    speaker_names = ""
    if speakers:
        speaker_names = "Speakers so far: "
        speaker_names += ", ".join([s.name for s in speakers])

    prompt = gen_speaker_prompt.format(speaker_names=speaker_names)

    messages = msg_helper(prompt, system=gen_speaker_system_prompt)
    response = await text_provider.chat_ollama(messages=messages, stream=False)
    return parse_json_content(response)


def parse_json_content(content: str | dict) -> dict:
    """
    since tool use can be iffy with the models I have been using, its worth trying to just ask the
    model to explicitly output the json info
    """

    # if pass in the raw response, extract the content
    if isinstance(content, dict):
        content = content.get("message", content).get("content", content)

    # check if the content has a ```json block which we should extract
    if "```json" in content:
        if json_match := re.search(r"```(?:json)?\n(.*?)\n```", content, re.DOTALL):  # type: ignore
            content = json_match.group(1)
        else:
            raise json.JSONDecodeError(f"Error with: '```json' block", doc=content, pos=0)  # type: ignore
    try:
        return json.loads(content)  # type: ignore
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"Invalid JSON content: {content}", doc=content, pos=0)  # type: ignore
